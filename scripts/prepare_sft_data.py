#!/usr/bin/env python3
"""Build a leakage-aware SkyRL SFT dataset from successful trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ATOMIC_TOOL_SCHEMA_SHAPES = {
    "glob": {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "include": {"type": "string"},
        },
        "required": ["pattern"],
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    },
    "localization_finish": {
        "type": "object",
        "properties": {
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "class_name": {"type": "string"},
                        "function_name": {"type": "string"},
                    },
                    "required": ["file"],
                },
            }
        },
        "required": ["locations"],
    },
}
ATOMIC_TOOL_NAMES = list(ATOMIC_TOOL_SCHEMA_SHAPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-instance-ids",
        type=Path,
        help="Optional text/JSON/JSONL file of instance IDs reserved for RL/eval.",
    )
    return parser.parse_args()


def load_excluded(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("instance_ids", value.keys())
        return {str(item) for item in value}
    excluded: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            excluded.add(str(row.get("instance_id", row.get("id"))))
        else:
            excluded.add(line)
    excluded.discard("None")
    return excluded


def is_chat(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    return len(roles) == len(messages) and "assistant" in roles


def schema_shape(schema: Any) -> dict | None:
    """Keep only the structural fields that define the model action space."""
    if not isinstance(schema, dict) or not isinstance(schema.get("type"), str):
        return None

    shape: dict[str, Any] = {"type": schema["type"]}
    if schema["type"] == "object":
        properties = schema.get("properties")
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        shaped_properties = {
            name: schema_shape(value) for name, value in properties.items()
        }
        if any(value is None for value in shaped_properties.values()):
            return None
        shape["properties"] = shaped_properties
        shape["required"] = required
    elif schema["type"] == "array":
        items = schema_shape(schema.get("items"))
        if items is None:
            return None
        shape["items"] = items
    return shape


def has_atomic_tool_schema(tools: Any) -> bool:
    if not isinstance(tools, list) or len(tools) != len(ATOMIC_TOOL_NAMES):
        return False
    for tool, expected_name in zip(tools, ATOMIC_TOOL_NAMES, strict=True):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return False
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != expected_name:
            return False
        if (
            schema_shape(function.get("parameters"))
            != ATOMIC_TOOL_SCHEMA_SHAPES[expected_name]
        ):
            return False
    return True


def normalized_row(path: Path, excluded: set[str], min_reward: float) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    instance_id = str(data.get("instance_id", ""))
    if not instance_id or instance_id in excluded:
        return None
    if float(data.get("total_reward", 0.0)) < min_reward:
        return None
    messages = data.get("sft_messages")
    if not is_chat(messages):
        # Also accept externally collected OpenAI-style trajectory files.
        messages = data.get("messages")
    if not is_chat(messages):
        return None
    tools = data.get("tools", [])
    if not has_atomic_tool_schema(tools):
        return None
    return {
        "instance_id": instance_id,
        "messages": messages,
        "tools": tools,
        "reward": float(data.get("total_reward", 0.0)),
        "source": str(path),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    excluded = load_excluded(args.exclude_instance_ids)
    candidates = []
    for path in sorted(args.trajectories.rglob("*.json")):
        row = normalized_row(path, excluded, args.min_reward)
        if row is not None:
            candidates.append(row)

    # Keep the best copy of duplicate conversations.
    deduped: dict[str, dict] = {}
    for row in candidates:
        digest = hashlib.sha256(
            json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if digest not in deduped or row["reward"] > deduped[digest]["reward"]:
            deduped[digest] = row
    rows = list(deduped.values())
    random.Random(args.seed).shuffle(rows)

    validation_size = round(len(rows) * args.validation_ratio)
    if args.validation_ratio > 0 and len(rows) > 1:
        validation_size = max(1, min(validation_size, len(rows) - 1))
    validation = rows[:validation_size]
    train = rows[validation_size:]
    if not train:
        raise RuntimeError("No usable SFT trajectories were found")

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "validation.jsonl", validation)
    report = {
        "scanned_json": len(list(args.trajectories.rglob("*.json"))),
        "eligible_before_dedup": len(candidates),
        "excluded_instance_ids": len(excluded),
        "train": len(train),
        "validation": len(validation),
        "min_reward": args.min_reward,
        "seed": args.seed,
    }
    report_path = args.output.parent / f"{args.output.name}-report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
