"""Analysis helpers for reproducible CodePin serving benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

PrometheusSnapshot = dict[tuple[str, tuple[tuple[str, str], ...]], float]

_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)"
    r"(?:\s+\d+)?$"
)
_LABEL = re.compile(r'(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"])*")')

# Two-sided Student-t 0.975 quantiles for 1..30 degrees of freedom.
# Repeated experiments normally have n=3; a normal interval is too narrow there.
_T975 = (
    12.706205,
    4.302653,
    3.182446,
    2.776445,
    2.570582,
    2.446912,
    2.364624,
    2.306004,
    2.262157,
    2.228139,
    2.200985,
    2.178813,
    2.160369,
    2.144787,
    2.131450,
    2.119905,
    2.109816,
    2.100922,
    2.093024,
    2.085963,
    2.079614,
    2.073873,
    2.068658,
    2.063899,
    2.059539,
    2.055529,
    2.051831,
    2.048407,
    2.045230,
    2.042272,
)

QUALITY_FIELDS = ("quality", "file_f1", "class_f1", "function_f1")
TOOL_BEHAVIOR_MAX_FIELDS = (
    "tool_errors",
    "repeated_searches",
    "overlap_lines",
    "output_chars",
    "excess_output_chars",
    "num_tool_calls",
    "num_turns",
    "read_lines",
    "truncated_outputs",
    "tool_efficiency_cost",
)


def source_manifest(root: Path) -> dict:
    """Reproducible workload bytes; only root Git metadata is excluded.

    This is provenance, not the service's more inclusive cache-invalidation key.
    """
    entries = []
    pending = [(root, "")]
    while pending:
        parent, prefix = pending.pop()
        with os.scandir(parent) as iterator:
            for entry in iterator:
                if not prefix and entry.name == ".git":
                    continue
                row = {"path": prefix + entry.name}
                if entry.is_symlink():
                    row.update(kind="link", target=os.readlink(entry.path))
                elif entry.is_dir(follow_symlinks=False):
                    row["kind"] = "directory"
                    pending.append((Path(entry.path), row["path"] + "/"))
                elif entry.is_file(follow_symlinks=False):
                    with open(entry.path, "rb") as handle:
                        content = hashlib.file_digest(handle, "sha256").hexdigest()
                    info = entry.stat(follow_symlinks=False)
                    row.update(
                        kind="file",
                        sha256=content,
                        bytes=info.st_size,
                        executable=bool(info.st_mode & 0o111),
                    )
                else:
                    raise ValueError(f"unsupported workload entry: {entry.path}")
                entries.append(row)
    entries.sort(key=lambda row: row["path"])
    return {
        "sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_files": sum(row["kind"] == "file" for row in entries),
        "source_bytes": sum(row.get("bytes", 0) for row in entries),
        "entries": entries,
    }


def has_runtime_exception(record: dict) -> bool:
    """Include typed server-side failures as well as client transport exceptions."""
    return bool(record.get("exception_type")) or any(
        error == "repository_or_deployment_changed_during_run"
        or re.match(r"^[\w.]*(?:Error|Exception|ExceptionGroup):", error)
        for error in record.get("errors", [])
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for ``fraction`` in [0, 1]."""
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    )


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize observations without hiding failed or missing samples."""
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stdev": None,
            "ci95_half_width": None,
            "p50": None,
            "p95": None,
        }
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    critical = _T975[len(samples) - 2] if 2 <= len(samples) <= 31 else 1.96
    return {
        "count": len(samples),
        "min": min(samples),
        "max": max(samples),
        "mean": statistics.mean(samples),
        "stdev": stdev,
        "ci95_half_width": (
            critical * stdev / math.sqrt(len(samples)) if len(samples) > 1 else None
        ),
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95),
    }


def _quality_value(record: dict[str, Any], field: str) -> float:
    try:
        value = float(
            record[field] if field == "quality" else record["quality_metrics"][field]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid quality measurement: {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite quality measurement: {field}")
    return value


def _tool_behavior_value(record: dict[str, Any], field: str) -> float:
    try:
        value = float(record["metrics"][field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid tool measurement: {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite tool measurement: {field}")
    return value


def summarize_task_behavior(
    records: Iterable[dict[str, Any]],
    *,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
    maximum_fields: Sequence[str] = TOOL_BEHAVIOR_MAX_FIELDS,
) -> dict[str, dict[str, Any]]:
    """Summarize quality floors and tool-cost ceilings for each real task."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if type(record.get("effective")) is not bool:
            raise ValueError("task behavior requires a boolean effective outcome")
        grouped.setdefault(str(record["instance_id"]), []).append(record)
    return {
        task: {
            "records": len(rows),
            "effective_rate": sum(bool(row.get("effective")) for row in rows)
            / len(rows),
            "minimum_quality": {
                field: min(_quality_value(row, field) for row in rows)
                for field in quality_fields
            },
            "maximum_tool_cost": {
                field: max(_tool_behavior_value(row, field) for row in rows)
                for field in maximum_fields
            },
        }
        for task, rows in sorted(grouped.items())
    }


def build_task_behavior_reference(
    record_runs: Iterable[Iterable[dict[str, Any]]],
    *,
    quality_fields: Sequence[str] = QUALITY_FIELDS,
    maximum_fields: Sequence[str] = TOOL_BEHAVIOR_MAX_FIELDS,
) -> dict[str, Any]:
    """Freeze per-task gates from independent accepted reference runs."""
    runs = [
        summarize_task_behavior(
            records,
            quality_fields=quality_fields,
            maximum_fields=maximum_fields,
        )
        for records in record_runs
    ]
    if not runs or not runs[0]:
        raise ValueError("task behavior reference requires nonempty records")
    task_set = set(runs[0])
    if any(set(run) != task_set for run in runs[1:]):
        raise ValueError("task behavior reference runs must contain the same tasks")
    return {
        "quality_fields": list(quality_fields),
        "maximum_fields": list(maximum_fields),
        "per_task": {
            task: {
                "minimum_effective_rate": min(
                    run[task]["effective_rate"] for run in runs
                ),
                "minimum_quality": {
                    field: min(run[task]["minimum_quality"][field] for run in runs)
                    for field in quality_fields
                },
                "maximum_tool_cost": {
                    field: max(run[task]["maximum_tool_cost"][field] for run in runs)
                    for field in maximum_fields
                },
                "records_per_reference_run": [run[task]["records"] for run in runs],
            }
            for task in sorted(task_set)
        },
    }


def evaluate_task_behavior(
    records: Iterable[dict[str, Any]],
    reference: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Evaluate one run against a frozen per-task quality and tool-cost gate."""
    quality_fields = tuple(reference["quality_fields"])
    maximum_fields = tuple(reference["maximum_fields"])
    actual = summarize_task_behavior(
        records,
        quality_fields=quality_fields,
        maximum_fields=maximum_fields,
    )
    expected = reference["per_task"]
    actual_tasks, expected_tasks = set(actual), set(expected)
    regressions: list[dict[str, Any]] = [
        {"task": task, "field": "missing_task"}
        for task in sorted(expected_tasks - actual_tasks)
    ]
    regressions.extend(
        {"task": task, "field": "unexpected_task"}
        for task in sorted(actual_tasks - expected_tasks)
    )
    for task in sorted(expected_tasks & actual_tasks):
        observed, gate = actual[task], expected[task]
        if observed["effective_rate"] < gate["minimum_effective_rate"] - tolerance:
            regressions.append(
                {
                    "task": task,
                    "field": "effective_rate",
                    "expected_minimum": gate["minimum_effective_rate"],
                    "actual": observed["effective_rate"],
                }
            )
        for field in quality_fields:
            value = observed["minimum_quality"][field]
            limit = gate["minimum_quality"][field]
            if value < limit - tolerance:
                regressions.append(
                    {
                        "task": task,
                        "field": field,
                        "expected_minimum": limit,
                        "actual": value,
                    }
                )
        for field in maximum_fields:
            value = observed["maximum_tool_cost"][field]
            limit = gate["maximum_tool_cost"][field]
            if value > limit + tolerance:
                regressions.append(
                    {
                        "task": task,
                        "field": field,
                        "expected_maximum": limit,
                        "actual": value,
                    }
                )
    return {
        "accepted": not regressions,
        "regressions": regressions,
        "per_task": actual,
    }


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Count equal tokens from the start of two token sequences."""
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _tool_delays(events: list[dict[str, Any]], turns: int) -> list[float]:
    actions: list[dict[str, Any]] = [
        event for event in events if event.get("kind") == "ActionEvent"
    ]
    observations = {
        event.get("tool_call_id"): event
        for event in events
        if event.get("kind") == "ObservationEvent"
    }
    groups: list[list[dict[str, Any]]] = []
    for action in actions:
        if not groups or groups[-1][0].get("llm_response_id") != action.get(
            "llm_response_id"
        ):
            groups.append([])
        groups[-1].append(action)
    delays = []
    for group in groups[:turns]:
        starts = [_timestamp(action["timestamp"]) for action in group]
        ends = [
            _timestamp(observations[action["tool_call_id"]]["timestamp"])
            for action in group
            if action.get("tool_call_id") in observations
        ]
        delays.append(
            max(0.0, (max(ends) - min(starts)).total_seconds()) if ends else 0.0
        )
    return (delays + [0.0] * turns)[:turns]


def analyze_token_trajectories(
    trajectories: Iterable[dict[str, Any]], *, cache_block_size: int
) -> dict[str, Any]:
    """Verify append-only prompts and estimate block-aligned cache opportunity."""
    if cache_block_size < 1:
        raise ValueError("cache_block_size must be positive")
    task_reports = []
    first_prompts: list[tuple[str, list[int]]] = []
    for trajectory in trajectories:
        instance_id = str(trajectory.get("instance_id") or "unknown")
        events = trajectory.get("messages") or []
        token_events = [event for event in events if event.get("kind") == "TokenEvent"]
        rounds = []
        delays = _tool_delays(events, len(token_events))
        for index, event in enumerate(token_events):
            prompt = list(event.get("prompt_token_ids") or [])
            response = list(event.get("response_token_ids") or [])
            report: dict[str, Any] = {
                "turn": index,
                "prompt_tokens": len(prompt),
                "response_tokens": len(response),
                "tool_duration_seconds": delays[index],
            }
            if index:
                previous = token_events[index - 1]
                previous_prompt = list(previous.get("prompt_token_ids") or [])
                previous_response = list(previous.get("response_token_ids") or [])
                computed = previous_prompt + previous_response
                prompt_lcp = common_prefix_length(previous_prompt, prompt)
                computed_lcp = common_prefix_length(computed, prompt)
                # The last sampled token need not have been fed back through
                # the model. Also leave one input token for the next logits.
                # This is a conservative opportunity estimate, not a hit count.
                reusable = min(
                    computed_lcp,
                    max(0, len(computed) - 1),
                    max(0, len(prompt) - 1),
                )
                report.update(
                    previous_prompt_is_prefix=prompt_lcp == len(previous_prompt),
                    previous_generation_is_prefix=computed_lcp == len(computed),
                    common_prefix_tokens=computed_lcp,
                    appended_observation_tokens=max(0, len(prompt) - len(computed)),
                    block_aligned_reusable_tokens=(
                        reusable // cache_block_size * cache_block_size
                    ),
                )
            rounds.append(report)
            if index == 0:
                first_prompts.append((instance_id, prompt))
        task_reports.append(
            {
                "instance_id": instance_id,
                "status": trajectory.get("status"),
                "turns": len(rounds),
                "tool_duration_total_seconds": sum(delays),
                "strict_append_only": bool(rounds)
                and all(
                    item["prompt_tokens"] and item["response_tokens"] for item in rounds
                )
                and all(
                    item.get("previous_generation_is_prefix", True) for item in rounds
                ),
                "rounds": rounds,
            }
        )

    pairs = []
    for index, (left_id, left) in enumerate(first_prompts):
        for right_id, right in first_prompts[index + 1 :]:
            prefix = common_prefix_length(left, right)
            pairs.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "common_prefix_tokens": prefix,
                    "block_aligned_reusable_tokens": (
                        min(prefix, max(0, min(len(left), len(right)) - 1))
                        // cache_block_size
                        * cache_block_size
                    ),
                }
            )
    within_task_rounds = [
        round_
        for task in task_reports
        for round_ in task["rounds"]
        if "common_prefix_tokens" in round_
    ]
    all_rounds = [round_ for task in task_reports for round_ in task["rounds"]]
    return {
        "cache_block_size": cache_block_size,
        "tasks": task_reports,
        "all_tasks_strict_append_only": bool(task_reports)
        and all(task["strict_append_only"] for task in task_reports),
        "cross_task_first_turn_pairs": pairs,
        "cross_task_common_prefix_tokens": summarize(
            pair["common_prefix_tokens"] for pair in pairs
        ),
        "cross_task_block_aligned_reusable_tokens": summarize(
            pair["block_aligned_reusable_tokens"] for pair in pairs
        ),
        "turns_per_task": summarize(task["turns"] for task in task_reports),
        "prompt_tokens_per_request": summarize(
            round_["prompt_tokens"] for round_ in all_rounds
        ),
        "response_tokens_per_request": summarize(
            round_["response_tokens"] for round_ in all_rounds
        ),
        "tool_duration_seconds": summarize(
            round_["tool_duration_seconds"] for round_ in all_rounds
        ),
        "within_task_common_prefix_tokens": summarize(
            round_["common_prefix_tokens"] for round_ in within_task_rounds
        ),
        "within_task_block_aligned_reusable_tokens": summarize(
            round_["block_aligned_reusable_tokens"] for round_ in within_task_rounds
        ),
    }


def build_replay_workload(
    trajectories: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract fixed token requests and observed tool delays from trajectories."""
    workload = []
    for trajectory in trajectories:
        events = trajectory.get("messages") or []
        token_events = [event for event in events if event.get("kind") == "TokenEvent"]
        if not token_events or any(
            not event.get("prompt_token_ids") or not event.get("response_token_ids")
            for event in token_events
        ):
            raise ValueError(
                "fixed replay requires nonempty real prompt and response tokens"
            )
        delays = _tool_delays(events, len(token_events))
        workload.append(
            {
                "instance_id": str(trajectory.get("instance_id") or "unknown"),
                "rounds": [
                    {
                        "turn": index,
                        "prompt_token_ids": list(event.get("prompt_token_ids") or []),
                        "response_tokens": len(event.get("response_token_ids") or []),
                        "tool_duration_seconds": delays[index],
                    }
                    for index, event in enumerate(token_events)
                ],
            }
        )
    return workload


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    """Load trajectory JSON files from a file or directory in stable order."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def parse_prometheus(text: str) -> PrometheusSnapshot:
    """Parse numeric samples from Prometheus' text exposition format."""
    samples: PrometheusSnapshot = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        labels = tuple(
            sorted(
                (key, json.loads(value))
                for key, value in _LABEL.findall(match.group("labels") or "")
            )
        )
        value = float(match.group("value"))
        samples[(match.group("name"), labels)] = value
    return samples


def metric_value(
    snapshot: PrometheusSnapshot,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Sum matching label variants of one metric name."""
    required = labels or {}
    return sum(
        value
        for (sample_name, sample_labels), value in snapshot.items()
        if sample_name == name
        and all(
            dict(sample_labels).get(key) == expected
            for key, expected in required.items()
        )
    )


def metric_delta(
    before: PrometheusSnapshot,
    after: PrometheusSnapshot,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Return a nonnegative counter delta across matching label sets."""
    return max(
        0.0,
        metric_value(after, name, labels) - metric_value(before, name, labels),
    )


def histogram_quantile_delta(
    before: PrometheusSnapshot,
    after: PrometheusSnapshot,
    name: str,
    fraction: float,
) -> float | None:
    """Estimate a histogram quantile from bucket counter deltas."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    bucket_name = name + "_bucket"
    boundaries: dict[float, float] = {}
    for (sample_name, labels), after_value in after.items():
        if sample_name != bucket_name:
            continue
        label_dict = dict(labels)
        if "le" not in label_dict:
            continue
        boundary = float(label_dict["le"])
        before_value = before.get((sample_name, labels), 0.0)
        boundaries[boundary] = boundaries.get(boundary, 0.0) + max(
            0.0, after_value - before_value
        )
    if not boundaries:
        return None
    total = max(boundaries.values())
    if total <= 0:
        return None
    target = total * fraction
    for boundary, count in sorted(boundaries.items()):
        if count >= target:
            return boundary
    return None
