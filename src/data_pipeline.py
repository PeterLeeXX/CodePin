"""Clean tasks, generate real trajectories, and export quality-filtered SkyRL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.build_swe_smith_code_search import OUTPUT_SCHEMA, BuildError, parse_patch
from src.trajectory import score_trajectory, validate_events
from src.utils.trajectory_tokens import build_assistant_loss_mask


def load_rows(path: Path) -> list[dict]:
    if path.is_dir():
        return [row for file in sorted(path.glob("*.json")) for row in load_rows(file)]
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else [value]


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def task_key(task: dict) -> str:
    return fingerprint(
        [
            task["repo"],
            task.get("base_commit"),
            " ".join(task["problem_statement"].split()),
            (task.get("patch") or "").replace("\r\n", "\n").strip(),
            bool(task.get("use_patch")),
        ]
    )


def difficulty(task: dict) -> str:
    changes = task["file_changes"]
    symbols = {
        symbol
        for file in changes
        for key in ("edited_entities", "added_entities")
        for symbol in ((file.get("changes") or {}).get(key) or [])
    }
    if len(changes) == 1 and len(symbols) <= 1:
        return "easy"
    return "medium" if len(changes) <= 3 and len(symbols) <= 5 else "hard"


def clean_tasks(rows: list[dict]) -> tuple[list[dict], dict]:
    from src.tools.localization_finish import CodeLocation

    kept, rejected = [], []
    seen_ids, seen_tasks = set(), set()
    counts = Counter()
    for row in rows:
        if not isinstance(row, dict):
            rejected.append({"instance_id": None, "reason": "invalid_task_object"})
            continue
        try:
            task = dict(row)
            for key in ("instance_id", "repo", "problem_statement"):
                if not isinstance(task.get(key), str) or not task[key].strip():
                    raise ValueError(f"empty_{key}")
                task[key] = task[key].strip()
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", task["repo"]):
                raise ValueError("invalid_repo")
            if any(part in {".", ".."} for part in task["repo"].split("/")):
                raise ValueError("invalid_repo")
            if not re.fullmatch(r"[\w.-]+", task["instance_id"]):
                raise ValueError("invalid_instance_id")
            if task.get("base_commit") and not re.fullmatch(
                r"[a-fA-F0-9]{40}", task["base_commit"]
            ):
                raise ValueError("base_commit_must_be_immutable")
            if "use_patch" in task and type(task["use_patch"]) is not bool:
                raise ValueError("use_patch_must_be_boolean")
            changes = task.get("file_changes")
            if not isinstance(changes, list) or not changes:
                raise ValueError("missing_targets")
            paths = []
            for change in changes:
                path = CodeLocation(file=change["file"]).file
                paths.append(path)
                details = change.get("changes") or {}
                if not isinstance(details, dict):
                    raise TypeError("invalid_target_details")
                for key, symbols in details.items():
                    if key not in {
                        "edited_modules",
                        "added_modules",
                        "edited_entities",
                        "added_entities",
                    }:
                        raise ValueError("unknown_target_field")
                    if symbols is not None and not isinstance(symbols, list):
                        raise ValueError("invalid_symbol_list")
                    if any(
                        not isinstance(s, str)
                        or not s.startswith(path + ":")
                        or not s[len(path) + 1 :]
                        for s in symbols or []
                    ):
                        raise ValueError("invalid_symbol_target")
            if len(set(paths)) != len(paths):
                raise ValueError("duplicate_target_file")
            if task.get("use_patch"):
                patch = parse_patch(task.get("patch", ""))
                if not patch or any(
                    f.is_added_file or f.is_removed_file for f in patch
                ):
                    raise ValueError("invalid_mutation_patch")
                if not set(paths).issubset({f.path for f in patch}):
                    raise ValueError("target_patch_mismatch")
            task["prompt"] = [{"role": "user", "content": task["problem_statement"]}]
            task["target"] = changes
            key = task_key(task)
            if task["instance_id"] in seen_ids or key in seen_tasks:
                raise ValueError("duplicate_task")
            seen_ids.add(task["instance_id"])
            seen_tasks.add(key)
            task["task_key"] = key
            task["difficulty"] = difficulty(task)
            kept.append(task)
            counts[task["difficulty"]] += 1
        except (BuildError, ValueError, TypeError, KeyError) as exc:
            reason = str(exc)
            rejected.append({"instance_id": row.get("instance_id"), "reason": reason})
    return kept, {
        "input": len(rows),
        "kept": len(kept),
        "rejected": rejected,
        "difficulty": dict(counts),
    }


def validate_trajectory(row: dict) -> list[str]:
    locations, errors = validate_events(row.get("messages", []))
    if errors:
        return errors
    if row.get("errors") or row.get("status") not in {"ok", "stop"}:
        errors.append("failed_run")
    if locations != row.get("structured_locations"):
        errors.append("finish_payload_mismatch")
    tokens = [m for m in row.get("messages", []) if m.get("kind") == "TokenEvent"]
    try:
        build_assistant_loss_mask(tokens)
    except (ValueError, KeyError, TypeError):
        errors.append("invalid_token_trace")
    messages = row.get("sft_messages", [])
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        return ["invalid_chat_trace"]
    pending: set[str] = set()
    calls = []
    chat_actions = []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            return ["invalid_chat_role"]
        if message.get("role") == "assistant":
            if pending:
                errors.append("missing_tool_result")
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                return ["invalid_chat_tool_calls"]
            for call in tool_calls:
                if not isinstance(call, dict) or not isinstance(
                    call.get("function"), dict
                ):
                    return ["invalid_chat_tool_calls"]
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in calls:
                    return ["invalid_chat_tool_ids"]
                pending.add(call_id)
                calls.append(call_id)
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise TypeError("arguments must be an object")
                    chat_actions.append((call["function"]["name"], arguments))
                except (ValueError, KeyError, TypeError):
                    errors.append("invalid_tool_arguments")
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                errors.append("orphan_chat_tool_result")
            else:
                pending.discard(call_id)
    if pending or not calls or not row.get("tools"):
        errors.append("incomplete_chat_trace")
    action_ids = [
        m.get("tool_call_id")
        for m in row.get("messages", [])
        if m.get("kind") == "ActionEvent"
    ]
    if calls != action_ids:
        errors.append("chat_event_mismatch")
    if not errors:
        actions = [m for m in row["messages"] if m.get("kind") == "ActionEvent"]
        for (name, arguments), event in zip(chat_actions, actions, strict=True):
            if name != event["tool_name"]:
                errors.append("chat_action_mismatch")
            elif name != "localization_finish":
                expected = {
                    k: v
                    for k, v in event["action"].items()
                    if k != "kind" and v is not None
                }
                if {k: v for k, v in arguments.items() if v is not None} != expected:
                    errors.append("chat_action_mismatch")
    assistants = [m for m in messages if m.get("role") == "assistant"]
    final = assistants[-1].get("tool_calls", []) if assistants else []
    if (
        len(final) != 1
        or final[0].get("function", {}).get("name") != "localization_finish"
    ):
        errors.append("invalid_final_chat_turn")
    elif not errors:
        final_locations = json.loads(final[0]["function"]["arguments"]).get("locations")
        if final_locations != locations:
            # Pydantic may add explicit null optional fields to event actions.
            normalize = lambda values: [
                {k: v for k, v in x.items() if v is not None} for x in values
            ]
            if normalize(final_locations or []) != normalize(locations or []):
                errors.append("chat_finish_mismatch")
    return sorted(set(errors))


def text_messages(messages: list[dict]) -> list[dict]:
    """Normalize SDK text blocks to the scalar content SkyRL also accepts."""
    result = []
    for message in messages:
        message = dict(message)
        if isinstance(message.get("content"), list):
            if any(item.get("type") != "text" for item in message["content"]):
                raise ValueError("CodePin exports text-only trajectories")
            message["content"] = "\n".join(item["text"] for item in message["content"])
        result.append(message)
    return result


def split_name(task: dict, validation_fraction: float) -> str:
    # Repository grouping prevents mutated versions of the same source snapshot
    # leaking across splits. All trajectories of a task follow the same split.
    group = fingerprint(task["repo"].lower())
    return "validation" if int(group[:8], 16) / 2**32 < validation_fraction else "train"


def export_data(
    tasks: list[dict],
    trajectories: list[dict],
    output: Path,
    *,
    min_quality: float = 0.5,
    max_cost: float = 3.0,
    validation_fraction: float = 0.1,
) -> dict:
    if not 0 <= min_quality <= 3 or not 0 <= validation_fraction < 1 or max_cost < 0:
        raise ValueError("invalid quality, cost or split threshold")
    output.mkdir(parents=True, exist_ok=False)
    tasks, cleaning = clean_tasks(tasks)
    indexed = {t["instance_id"]: t for t in tasks}
    report = {"cleaning": cleaning, "sft_kept": 0, "rejected": [], "splits": {}}
    sft = {"train": [], "validation": []}
    seen = set()
    for row in trajectories:
        task = indexed.get(row.get("instance_id"))
        errors = validate_trajectory(row)
        if task is None:
            errors.append("unknown_task")
        elif not errors:
            _, scores, metrics = score_trajectory(
                task, row["structured_locations"], row["messages"]
            )
            if scores["multilevel_localization_f1_reward"] < min_quality:
                errors.append("low_localization_quality")
            if metrics["tool_efficiency_cost"] > max_cost:
                errors.append("excess_tool_cost")
        if errors:
            report["rejected"].append(
                {"instance_id": row.get("instance_id"), "reasons": errors}
            )
            continue
        # Tool IDs and absolute temporary paths vary between identical runs.
        actions = [
            (m.get("tool_name"), m.get("action"))
            for m in row.get("messages", [])
            if m.get("kind") == "ActionEvent"
        ]
        key = fingerprint([task_key(task) if task else None, actions])
        if key in seen:
            report["rejected"].append(
                {
                    "instance_id": row.get("instance_id"),
                    "reasons": ["duplicate_trajectory"],
                }
            )
            continue
        seen.add(key)
        prompt, response, mask = build_assistant_loss_mask(
            [m for m in row["messages"] if m.get("kind") == "TokenEvent"]
        )
        sft[split_name(task, validation_fraction)].append(
            {
                "instance_id": task["instance_id"],
                "difficulty": task["difficulty"],
                "messages": text_messages(row["sft_messages"]),
                "tools": json.dumps(row["tools"], ensure_ascii=False),
                "input_ids": prompt + response,
                "loss_mask": [0] * len(prompt) + mask,
            }
        )
        report["sft_kept"] += 1
    sft_schema = pa.schema(
        [
            ("instance_id", pa.string()),
            ("difficulty", pa.string()),
            (
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            ("role", pa.string()),
                            ("content", pa.string()),
                            ("tool_call_id", pa.string()),
                            ("name", pa.string()),
                            (
                                "tool_calls",
                                pa.list_(
                                    pa.struct(
                                        [
                                            ("id", pa.string()),
                                            ("type", pa.string()),
                                            (
                                                "function",
                                                pa.struct(
                                                    [
                                                        ("name", pa.string()),
                                                        ("arguments", pa.string()),
                                                    ]
                                                ),
                                            ),
                                        ]
                                    )
                                ),
                            ),
                        ]
                    )
                ),
            ),
            ("tools", pa.string()),
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int64())),
        ]
    )
    # Preserve actual immutable base commits; the legacy sample schema is null.
    fields = [
        pa.field(f.name, pa.string()) if f.name == "base_commit" else f
        for f in OUTPUT_SCHEMA
    ]
    rl_schema = pa.schema(fields + [pa.field("difficulty", pa.string())])
    for split, sft_rows in sft.items():
        rl = [t for t in tasks if split_name(t, validation_fraction) == split]
        for name, rows, schema in (
            ("sft", sft_rows, sft_schema),
            ("rl", rl, rl_schema),
        ):
            directory = output / name
            directory.mkdir(exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema),
                directory / f"{split}.parquet",
            )
            report["splits"][f"{name}/{split}"] = len(rows)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate_trajectories(
    tasks: list[dict],
    output: Path,
    *,
    model: str,
    base_url: str,
    concurrency: int = 4,
    max_turns: int = 8,
    max_tokens: int = 2048,
) -> list[dict]:
    from src.rollout import run_localization
    from src.utils.instance import clone_instance

    if not 1 <= concurrency <= 32:
        raise ValueError("concurrency must be 1..32")
    output.mkdir(parents=True, exist_ok=False)

    def run(task):
        with tempfile.TemporaryDirectory(prefix="codepin-data-") as directory:
            ok, root = clone_instance(
                task["repo"],
                task.get("base_commit"),
                task["instance_id"],
                Path(directory),
                task.get("patch") if task.get("use_patch") else None,
            )
            if not ok:
                result = {
                    "status": "error",
                    "errors": ["checkout_failed"],
                    "messages": [],
                }
            else:
                result = run_localization(
                    task,
                    root,
                    model=model,
                    base_url=base_url,
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                )
                reward, details, metrics = score_trajectory(
                    task,
                    result["structured_locations"],
                    result["messages"],
                    valid=result["status"] == "ok",
                )
                result.update(total_reward=reward, reward_dict=details)
                result["metrics"].update(metrics)
            result["instance_id"] = task["instance_id"]
            (output / f"{task['instance_id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result

    tasks, report = clean_tasks(tasks)
    (output / "cleaning_report.txt").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(run, tasks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("clean", "generate", "export"):
        child = sub.add_parser(command)
        child.add_argument("--tasks", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "generate":
            child.add_argument("--model", default="openai/codepin")
            child.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
            child.add_argument("--concurrency", type=int, default=4)
            child.add_argument("--max-turns", type=int, default=8)
            child.add_argument("--max-tokens", type=int, default=2048)
        if command == "export":
            child.add_argument("--trajectories", type=Path, required=True)
            child.add_argument("--min-quality", type=float, default=0.5)
            child.add_argument("--max-cost", type=float, default=3)
            child.add_argument("--validation-fraction", type=float, default=0.1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    tasks = load_rows(args.pop("tasks"))
    if command == "export":
        print(
            json.dumps(export_data(tasks, load_rows(args.pop("trajectories")), **args))
        )
    elif command == "generate":
        generate_trajectories(tasks, **args)
    else:
        rows, report = clean_tasks(tasks)
        output = args["output"]
        output.mkdir(parents=True, exist_ok=False)
        (output / "tasks.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
