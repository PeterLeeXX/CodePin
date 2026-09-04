"""Freeze repeated baseline behavior and compare complete real-task outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.report_performance import read_records
from src.performance import (
    QUALITY_FIELDS,
    TOOL_BEHAVIOR_MAX_FIELDS,
    build_task_behavior_reference,
    evaluate_task_behavior,
    has_runtime_exception,
    summarize,
    summarize_task_behavior,
)

CONTEXT_COST_FIELDS = {
    "output_chars",
    "read_lines",
    "excess_output_chars",
    "truncated_outputs",
    "tool_efficiency_cost",
}


def task_means(records: list[dict]) -> dict:
    """Validate every measurement before computing equal-task comparisons."""
    summarize_task_behavior(records)
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["instance_id"])].append(record)
    return {
        task: {
            "quality": {
                key: sum(
                    row["quality"] if key == "quality" else row["quality_metrics"][key]
                    for row in rows
                )
                / len(rows)
                for key in QUALITY_FIELDS
            },
            "tool_cost": {
                key: sum(row["metrics"][key] for row in rows) / len(rows)
                for key in TOOL_BEHAVIOR_MAX_FIELDS
            },
        }
        for task, rows in sorted(grouped.items())
    }


def measured_records(records: list[dict]) -> list[dict]:
    """Keep real trajectory measurements separate from infrastructure failures."""
    for row in records:
        if type(row.get("effective")) is not bool:
            raise ValueError("every submitted task requires an effective outcome")
        if has_runtime_exception(row) and row["effective"]:
            raise ValueError("an infrastructure failure cannot be effective")
    measured = [row for row in records if not has_runtime_exception(row)]
    summarize_task_behavior(measured)
    return measured


def outcome_counts(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in records:
        grouped[str(row["instance_id"])].append(row)
    return {
        task: {
            "submitted": len(rows),
            "effective": sum(row["effective"] for row in rows),
            "effective_rate": sum(row["effective"] for row in rows) / len(rows),
            "infrastructure_failures": sum(has_runtime_exception(row) for row in rows),
        }
        for task, rows in sorted(grouped.items())
    }


def freeze_reference(
    record_runs: list[list[dict]], *, allow_quality_improving_context: bool = False
) -> dict:
    if len(record_runs) < 3:
        raise ValueError("at least three independent baseline runs are required")
    measured_runs = [measured_records(records) for records in record_runs]
    for records, measured in zip(record_runs, measured_runs, strict=True):
        if {str(row["instance_id"]) for row in records} != {
            str(row["instance_id"]) for row in measured
        }:
            raise ValueError("every baseline task requires measured trajectories")
    strict = build_task_behavior_reference(measured_runs)
    means = [task_means(records) for records in measured_runs]
    reference = {
        "strict_reference": strict,
        "all_baseline_outcome_counts": [
            outcome_counts(records) for records in record_runs
        ],
        "run_mean_reference": {
            task: {
                section: {
                    field: summarize(run[task][section][field] for run in means)
                    for field in fields
                }
                for section, fields in (
                    ("quality", QUALITY_FIELDS),
                    ("tool_cost", TOOL_BEHAVIOR_MAX_FIELDS),
                )
            }
            for task in strict["per_task"]
        },
        "interpretation": (
            "Empirical baseline envelope, not a statistical non-inferiority proof. "
            "Retain per-task effective-rate floors and individual quality/cost extrema; "
            "add mean quality floors and mean tool-cost ceilings from independent "
            "baseline run means. All candidate runs must stay within those observed "
            "bounds. CI describes run variation; it does not enlarge the bounds. "
            "Infrastructure failures stay in all-outcome performance and quality "
            "statistics; they cannot lower conditional localization/tool bounds. "
            "Candidates must have zero infrastructure failures. Model/finish/tool "
            "failures with actual trajectory measurements remain in those bounds. "
            "Effective-task definitions and historical screen results are unchanged."
        ),
    }
    if allow_quality_improving_context:
        reference["context_cost_policy"] = "quality_improvement"
        reference["best_baseline_quality"] = {
            task: {
                field: max(
                    row["quality"]
                    if field == "quality"
                    else row["quality_metrics"][field]
                    for records in measured_runs
                    for row in records
                    if str(row["instance_id"]) == task
                )
                for field in QUALITY_FIELDS
            }
            for task in strict["per_task"]
        }
        reference["interpretation"] += (
            " Opt-in context-cost policy: higher read/output costs may be reported as "
            "a quality/cost tradeoff only for valid, tool-error-free outcomes whose "
            "localization vector dominates every observed baseline quality maximum "
            "and strictly improves at least one F1 level. Ordinary-outcome cost means "
            "must still satisfy baseline bounds. All raw costs, scores and effective "
            "outcomes remain unchanged. Quality, misuse, turn/call and infrastructure "
            "gates remain strict. This does not prove every observed line is useful."
        )
    return reference


def context_cost_tradeoffs(
    records: list[dict], reference: dict, regressions: list[dict]
):
    """Distinguish improved localization with more context from unchanged-quality waste."""
    grouped = defaultdict(list)
    for row in records:
        grouped[str(row["instance_id"])].append(row)
    improved, ordinary = {}, {}
    for task, rows in grouped.items():
        best = reference["best_baseline_quality"].get(task)
        if best is None:
            continue
        improved[task], ordinary[task] = [], []
        for row in rows:
            quality = {
                field: row["quality"]
                if field == "quality"
                else row["quality_metrics"][field]
                for field in QUALITY_FIELDS
            }
            useful = (
                row.get("status") == "ok"
                and not row.get("errors")
                and row["metrics"]["tool_errors"] == 0
                and all(
                    quality[field] >= best[field] - 1e-8 for field in QUALITY_FIELDS
                )
                and any(
                    quality[field] > best[field] + 1e-8
                    for field in QUALITY_FIELDS
                    if field != "quality"
                )
            )
            (improved[task] if useful else ordinary[task]).append(row)
    remaining, tradeoffs = [], []
    for regression in regressions:
        task = regression["task"]
        field = regression["field"].removeprefix("mean_")
        eligible = field in CONTEXT_COST_FIELDS and bool(improved.get(task))
        ordinary_mean = None
        if eligible and regression["field"].startswith("mean_"):
            values = [row["metrics"][field] for row in ordinary[task]]
            ordinary_mean = sum(values) / len(values) if values else None
            eligible = (
                ordinary_mean is None
                or ordinary_mean <= regression["expected_max"] + 1e-8
            )
        elif eligible:
            limit = regression["expected_maximum"]
            eligible = all(
                row["metrics"][field] <= limit + 1e-8 for row in ordinary[task]
            )
        if eligible:
            tradeoffs.append(
                {
                    **regression,
                    "improved_quality_outcomes": len(improved[task]),
                    "ordinary_outcomes": len(ordinary[task]),
                    "ordinary_cost_mean": ordinary_mean,
                    "best_baseline_quality": reference["best_baseline_quality"][task],
                    "interpretation": "Higher context cost with improved measured localization; raw cost and effective-task definition unchanged.",
                }
            )
        else:
            remaining.append(regression)
    return remaining, tradeoffs


def compare_records(records: list[dict], reference: dict) -> dict:
    measured = measured_records(records)
    result = evaluate_task_behavior(measured, reference["strict_reference"])
    counts = outcome_counts(records)
    for task, values in counts.items():
        if values["infrastructure_failures"]:
            result["regressions"].append(
                {
                    "task": task,
                    "field": "infrastructure_failures",
                    "expected_max": 0,
                    "actual": values["infrastructure_failures"],
                }
            )
    means = task_means(measured)
    expected = reference["run_mean_reference"]
    for task in sorted(means.keys() & expected.keys()):
        for section, bound in (("quality", "min"), ("tool_cost", "max")):
            for field, statistics in expected[task][section].items():
                actual, limit = means[task][section][field], statistics[bound]
                regressed = (
                    actual < limit - 1e-8 if bound == "min" else actual > limit + 1e-8
                )
                if regressed:
                    result["regressions"].append(
                        {
                            "task": task,
                            "field": f"mean_{field}",
                            "expected_" + bound: limit,
                            "actual": actual,
                        }
                    )
    result["run_means"] = means
    result["all_outcome_counts"] = counts
    if reference.get("context_cost_policy") == "quality_improvement":
        result["regressions"], result["context_cost_tradeoffs"] = (
            context_cost_tradeoffs(measured, reference, result["regressions"])
        )
    result["accepted"] = not result["regressions"]
    return result


def load_measured_run(path: Path) -> tuple[list[dict], dict]:
    records_path = path / "records.jsonl"
    records = read_records(records_path)
    summary = json.loads((path / "summary.json").read_text())
    if not records or len(records) != summary["submitted_tasks"]:
        raise ValueError("record count differs from submitted tasks")
    measured = measured_records(records)
    if any(row.get("cache_hit") is not False for row in measured):
        raise ValueError("all outcomes must explicitly disable result cache hits")
    if summary["vllm_metrics"]["vllm:request_success_total"] != sum(
        row.get("metrics", {}).get("num_turns", 0) for row in records
    ):
        raise ValueError("native model request count differs from trajectory turns")
    config = summary["config"]
    definition = {
        key: config[key]
        for key in (
            "max_turns",
            "max_tokens",
            "max_context_chars",
            "max_context_lines",
            "minimum_quality",
            "maximum_tool_errors",
            "require_context",
            "seed",
            "split",
        )
    }
    return records, {
        "path": str(path),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(
            (path / "summary.json").read_bytes()
        ).hexdigest(),
        "task_definition": definition,
        "all_outcome_statistics": {
            key: summary[key]
            for key in (
                "submitted_tasks",
                "effective_rate",
                "infrastructure_error_rate",
                "timeout_rate",
                "quality",
                "file_f1",
                "class_f1",
                "function_f1",
            )
        },
        "load": {
            key: config[key]
            for key in (
                "mcp_clients",
                "client_concurrency",
                "continuous",
                "cycles",
                "minimum_duration",
                "reset_prefix_before",
                "reset_prefix_between_cycles",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--allow-quality-improving-context", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        loaded = [load_measured_run(path) for path in args.baseline_runs]
        sources = [source for _, source in loaded]
        if len({str(path.resolve()) for path in args.baseline_runs}) != len(sources):
            raise ValueError("duplicate runs are not independent repetitions")
        if len({source["records_sha256"] for source in sources}) != len(sources):
            raise ValueError("copied run evidence is not an independent repetition")
        if any(
            source["task_definition"] != sources[0]["task_definition"]
            or source["load"] != sources[0]["load"]
            for source in sources
        ):
            raise ValueError(
                "baseline repetitions must use the same task definition and load"
            )
        result = freeze_reference(
            [records for records, _ in loaded],
            allow_quality_improving_context=args.allow_quality_improving_context,
        )
        result["sources"] = sources
    else:
        reference = json.loads(args.reference.read_text())
        records, source = load_measured_run(args.run)
        if source["task_definition"] != reference["sources"][0]["task_definition"]:
            raise ValueError("candidate changed the task definition")
        result = compare_records(records, reference)
        result["source"] = source
        result["reference_sha256"] = hashlib.sha256(
            args.reference.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "accepted": result.get("accepted")}))


if __name__ == "__main__":
    main()
