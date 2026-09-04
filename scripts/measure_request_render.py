"""Diagnose real chat serialization/rendering without GPU generation.

This is a separate diagnostic, never an end-to-end task benchmark. The native
render endpoint must reproduce every recorded prompt token exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from src.performance import load_trajectories, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with httpx.Client(timeout=30) as client:
        for trajectory in load_trajectories(args.trajectories):
            messages = trajectory["sft_messages"]
            tokens = [e for e in trajectory["messages"] if e["kind"] == "TokenEvent"]
            boundaries = [
                i
                for i, message in enumerate(messages)
                if message["role"] == "assistant"
            ]
            if not tokens or len(tokens) != len(boundaries):
                raise ValueError(
                    "real messages and TokenEvents must have matching turns"
                )
            for turn, (boundary, event) in enumerate(
                zip(boundaries, tokens, strict=True)
            ):
                body = {
                    "model": "codepin",
                    "messages": messages[:boundary],
                    "tools": trajectory["tools"],
                    "tool_choice": "auto",
                    "temperature": 0,
                    "max_tokens": 2048,
                    "chat_template_kwargs": {
                        "add_generation_prompt": True,
                        "enable_thinking": False,
                    },
                }
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    encoded = json.dumps(
                        body, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    serialization = time.perf_counter() - started
                    started = time.perf_counter()
                    response = client.post(
                        args.base_url + "/chat/completions/render",
                        content=encoded,
                        headers={"content-type": "application/json"},
                    )
                    response.raise_for_status()
                    rendered = response.json()
                    elapsed = time.perf_counter() - started
                    (args.output / "last-render.json").write_text(json.dumps(rendered))
                    actual = rendered["token_ids"]
                    matched = actual == event["prompt_token_ids"]
                    rows.append(
                        {
                            "instance_id": trajectory["instance_id"],
                            "turn": turn,
                            "repeat": repeat,
                            "request_bytes": len(encoded),
                            "prompt_tokens": len(actual),
                            "exact_tokens_match": matched,
                            "serialization_seconds": serialization,
                            "http_render_roundtrip_seconds": elapsed,
                        }
                    )
                    with (args.output / "records.jsonl").open("a") as stream:
                        stream.write(json.dumps(rows[-1]) + "\n")
                    if not matched:
                        raise ValueError(
                            "native render differs from actual inference prompt tokens"
                        )
    report = {
        "requests": len(rows),
        "all_tokens_match": all(r["exact_tokens_match"] for r in rows),
        "serialization_seconds": summarize(r["serialization_seconds"] for r in rows),
        "http_render_roundtrip_seconds": summarize(
            r["http_render_roundtrip_seconds"] for r in rows
        ),
        "interpretation": "Includes loopback HTTP, native chat parsing/template/tokenization and response JSON. No GPU inference or tool execution; diagnostic only.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
