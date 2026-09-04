"""Compare explicitly selected real runs, preserving failures and run-level variation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.performance import has_runtime_exception, summarize


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_task(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["instance_id"]].append(record)
    result = {}
    for name, rows in sorted(grouped.items()):
        result[name] = {
            "count": len(rows),
            "effective_rate": sum(bool(row.get("effective")) for row in rows)
            / len(rows),
            "quality": summarize(row.get("quality", 0) for row in rows),
            "quality_metrics": {
                level: summarize(
                    row.get("quality_metrics", {}).get(level, 0) for row in rows
                )
                for level in ("file_f1", "class_f1", "function_f1")
            },
            "latency_seconds": summarize(row["latency_seconds"] for row in rows),
            "tool_calls": summarize(
                row.get("metrics", {}).get("num_tool_calls", 0) for row in rows
            ),
            "output_chars": summarize(
                row.get("metrics", {}).get("output_chars", 0) for row in rows
            ),
            "max_prompt_tokens": summarize(
                row["metrics"]["max_prompt_tokens"]
                for row in rows
                if "max_prompt_tokens" in row.get("metrics", {})
            ),
            "model_turns": summarize(
                row.get("metrics", {}).get("num_turns", 0) for row in rows
            ),
            "completion_tokens": summarize(
                row.get("metrics", {}).get("completion_tokens", 0) for row in rows
            ),
            "errors": sorted(
                {error for row in rows for error in row.get("errors", [])}
            ),
            "locations": sorted(
                {json.dumps(row.get("locations", []), sort_keys=True) for row in rows}
            ),
            "behavior_values": {
                key: sorted({row.get("metrics", {}).get(key, 0) for row in rows})
                for key in (
                    "num_tool_calls",
                    "num_turns",
                    "repeated_searches",
                    "read_lines",
                    "overlap_lines",
                    "output_chars",
                    "excess_output_chars",
                    "truncated_outputs",
                    "tool_errors",
                    "tool_efficiency_cost",
                )
            },
        }
    return result


def scalar_metrics(summary: dict) -> dict:
    values = {
        key: summary[key]
        for key in (
            "effective_tasks_per_minute",
            "terminal_tasks_per_minute",
            "admitted_terminal_tasks_per_minute",
            "successful_tasks_per_minute",
            "success_rate",
            "effective_rate",
            "timeout_rate",
            "error_rate",
            "infrastructure_error_rate",
            "admission_rejection_rate",
            "mcp_startup_seconds",
        )
        if key in summary
    }
    for key in (
        "latency_seconds",
        "admitted_latency_seconds",
        "successful_latency_seconds",
    ):
        for statistic in ("mean", "p50", "p95"):
            if key in summary:
                values[f"{key}.{statistic}"] = summary[key][statistic]
    for key in ("quality", "file_f1", "class_f1", "function_f1"):
        if key in summary:
            values[key] = summary[key]["mean"]
    for section in ("resources", "tool_metrics", "stage_metrics"):
        for key, value in summary.get(section, {}).items():
            if value.get("mean") is not None:
                values[f"{section}.{key}"] = value["mean"]
    for key, value in summary.get("vllm_metrics", {}).items():
        if isinstance(value, int | float):
            values[f"vllm_metrics.{key}"] = value
        elif isinstance(value, dict) and "mean" in value:
            values[f"vllm_metrics.{key}.mean"] = value["mean"]
        elif key == "prompt_tokens_by_source":
            for source, count in value.items():
                values[f"vllm_metrics.{key}.{source}"] = count
                values[f"{source}_tokens_per_task"] = count / summary["submitted_tasks"]
    values["generated_tokens_per_second"] = (
        summary.get("vllm_metrics", {}).get("vllm:generation_tokens_total", 0)
        / summary["duration_seconds"]
    )
    return values


def steady_windows(run: Path, seconds: float = 60, steady_start: float = 60) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    rows = read_records(run / "records.jsonl")
    resources = read_records(run / "resources.jsonl")
    count = int(summary["duration_seconds"] // seconds)
    windows = []
    for index in range(count):
        start, end = index * seconds, (index + 1) * seconds
        completed = [
            row
            for row in rows
            if start <= row["submitted_offset_seconds"] + row["latency_seconds"] < end
        ]
        active = [row for row in resources if start <= row["elapsed_seconds"] < end]
        windows.append(
            {
                "start_seconds": start,
                "terminal_tasks_per_minute": len(completed) / seconds * 60,
                "effective_tasks_per_minute": sum(
                    bool(row.get("effective")) for row in completed
                )
                / seconds
                * 60,
                "latency_seconds": summarize(
                    row["latency_seconds"] for row in completed
                ),
                "resources": {
                    key: summarize(row[key] for row in active if key in row)
                    for key in (
                        "memory_anon_bytes",
                        "memory_file_bytes",
                        "memory_current_bytes",
                        "gpu_memory_used_bytes",
                        "gpu_utilization_percent",
                        "vllm_waiting_requests",
                        "vllm_running_requests",
                        "cpu_quota_utilization_percent",
                    )
                },
            }
        )
    return {
        "run": str(run),
        "complete_windows": windows,
        "total_seconds": summary["duration_seconds"],
        "steady_start_seconds": steady_start,
        "submission_end_seconds": summary["config"].get("minimum_duration", 0),
    }


def continuous_window(
    summary: dict, rows: list[dict], start: float, resources: list[dict] | None = None
) -> dict:
    """Use a fixed admission window; retain the later results of its whole cohort."""
    end = summary["config"].get("minimum_duration", 0)
    if not summary["config"].get("continuous") or end <= start:
        return {}
    completed = [
        row
        for row in rows
        if start <= row["submitted_offset_seconds"] + row["latency_seconds"] < end
    ]
    submitted = [row for row in rows if start <= row["submitted_offset_seconds"] < end]
    samples = [
        sample for sample in resources or [] if start <= sample["elapsed_seconds"] < end
    ]
    resource_keys = {
        key
        for sample in samples
        for key, value in sample.items()
        if key != "elapsed_seconds" and isinstance(value, int | float)
    }
    return {
        "steady_window_seconds": end - start,
        "steady_resource_sample_count": len(samples),
        "steady_metrics_scrape_failure_count": sum(
            bool(sample.get("metrics_scrape_failed")) for sample in samples
        ),
        "steady_terminal_count": len(completed),
        "steady_submitted_count": len(submitted),
        "steady_effective_tasks_per_minute": sum(
            bool(row["effective"]) for row in completed
        )
        / (end - start)
        * 60,
        "steady_terminal_tasks_per_minute": len(completed) / (end - start) * 60,
        "steady_admitted_terminal_tasks_per_minute": sum(
            not row.get("admission_rejected") for row in completed
        )
        / (end - start)
        * 60,
        "steady_successful_tasks_per_minute": sum(
            row.get("status") == "ok" for row in completed
        )
        / (end - start)
        * 60,
        "steady_latency_seconds.p95": summarize(
            row["latency_seconds"] for row in submitted
        )["p95"],
        "steady_latency_seconds.p50": summarize(
            row["latency_seconds"] for row in submitted
        )["p50"],
        "steady_admitted_latency_seconds.p95": summarize(
            row["latency_seconds"]
            for row in submitted
            if not row.get("admission_rejected")
        )["p95"],
        "steady_effective_rate": sum(bool(row["effective"]) for row in submitted)
        / len(submitted)
        if submitted
        else None,
        **{
            f"steady_resources.{key}": summarize(
                sample[key] for sample in samples if key in sample
            )["mean"]
            for key in resource_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.spec.read_text())
    groups = {}
    for group in spec["groups"]:
        summaries = [
            json.loads((args.results / name / "summary.json").read_text())
            for name in group["runs"]
        ]
        record_sets = [
            read_records(args.results / name / "records.jsonl")
            for name in group["runs"]
        ]
        resource_sets = [
            read_records(args.results / name / "resources.jsonl")
            for name in group["runs"]
        ]
        metrics = [
            scalar_metrics(summary)
            | continuous_window(
                summary, rows, spec.get("steady_start_seconds", 10), resources
            )
            | {
                "infrastructure_error_rate": sum(map(has_runtime_exception, rows))
                / len(rows)
            }
            for summary, rows, resources in zip(
                summaries, record_sets, resource_sets, strict=True
            )
        ]
        records = [row for rows in record_sets for row in rows]
        groups[group["name"]] = {
            **group,
            "metrics": {
                key: summarize(row[key] for row in metrics if row.get(key) is not None)
                for key in sorted(set().union(*metrics))
            },
            "per_task": per_task(records),
            "difficulty": {
                difficulty: {
                    "tasks": len(
                        selected := [
                            row for row in records if row["difficulty"] == difficulty
                        ]
                    ),
                    "effective_rate": sum(bool(row["effective"]) for row in selected)
                    / len(selected),
                    "latency_seconds": summarize(
                        row["latency_seconds"] for row in selected
                    ),
                }
                for difficulty in sorted({row["difficulty"] for row in records})
            },
            "run_scalars": metrics,
        }
    comparisons = []
    for pair in spec.get("comparisons", []):
        base, final = groups[pair["baseline"]], groups[pair["candidate"]]
        changes = {}
        for key in base["metrics"].keys() & final["metrics"].keys():
            before, after = base["metrics"][key]["mean"], final["metrics"][key]["mean"]
            changes[key] = {
                "absolute": after - before,
                "relative_percent": (after / before - 1) * 100 if before else None,
            }
        paired_changes = {}
        for key in pair.get("paired_metrics", []):
            paired = list(zip(base["run_scalars"], final["run_scalars"], strict=True))
            paired_changes[key] = {
                "absolute": summarize(
                    after[key] - before[key] for before, after in paired
                ),
                "relative_percent": summarize(
                    (after[key] / before[key] - 1) * 100
                    for before, after in paired
                    if before[key]
                ),
            }
        comparisons.append(
            {
                **pair,
                "changes": changes,
                "paired_changes": paired_changes,
                "per_task_effective_rate_delta": {
                    key: final["per_task"][key]["effective_rate"]
                    - value["effective_rate"]
                    for key, value in base["per_task"].items()
                    if key in final["per_task"]
                },
                "per_task_regression": {
                    key: {
                        "same_locations": final["per_task"][key]["locations"]
                        == value["locations"],
                        "same_behavior_values": final["per_task"][key][
                            "behavior_values"
                        ]
                        == value["behavior_values"],
                        "completion_tokens": {
                            "baseline": value["completion_tokens"],
                            "candidate": final["per_task"][key]["completion_tokens"],
                        },
                        "minimum_quality_delta": final["per_task"][key]["quality"][
                            "min"
                        ]
                        - value["quality"]["min"],
                        "per_level_f1_minimum_delta": {
                            level: final["per_task"][key]["quality_metrics"][level][
                                "min"
                            ]
                            - metric["min"]
                            for level, metric in value["quality_metrics"].items()
                            if metric["count"]
                            and final["per_task"][key]["quality_metrics"][level][
                                "count"
                            ]
                        },
                    }
                    for key, value in base["per_task"].items()
                    if key in final["per_task"]
                },
            }
        )
    report = {
        "spec": spec,
        "groups": groups,
        "comparisons": comparisons,
        "steady": [
            steady_windows(
                args.results / name, steady_start=spec.get("steady_start_seconds", 60)
            )
            for name in spec.get("steady_runs", [])
        ],
        "statistics": "Group confidence intervals use independent run scalars, Student-t for n<=31. Task percentiles describe all terminal outcomes. Steady buckets count task completions; overlapping stages are not summed into latency.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if args.plots:
        plot_report(report, args.output)
    print(
        json.dumps(
            {
                "groups": len(groups),
                "comparisons": len(comparisons),
                "output": str(args.output),
            }
        )
    )


def plot_report(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    series = defaultdict(list)
    for group in report["groups"].values():
        if "load" in group and "series" in group:
            series[group["series"]].append(group)
    if series:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        throughput_key = report["spec"].get(
            "load_throughput_metric", "effective_tasks_per_minute"
        )
        latency_key = report["spec"].get("load_latency_metric", "latency_seconds.p95")
        for name, groups in series.items():
            groups.sort(key=lambda group: group["load"])
            for axis, key in zip(axes, (throughput_key, latency_key), strict=True):
                axis.errorbar(
                    [group["load"] for group in groups],
                    [group["metrics"][key]["mean"] for group in groups],
                    yerr=[group["metrics"][key]["stdev"] for group in groups],
                    marker="o",
                    label=name,
                    capsize=3,
                )
                axis.set(
                    xlabel=report["spec"].get(
                        "load_axis_label", "Concurrent end-to-end tasks"
                    ),
                    ylabel="Effective tasks/min"
                    if key == throughput_key
                    else report["spec"].get(
                        "load_latency_label", "P95 end-to-end latency (s)"
                    ),
                )
                axis.grid(alpha=0.2)
        failed = [
            group
            for groups in series.values()
            for group in groups
            if any(
                group["metrics"].get(key, {}).get("mean", 0) > 0
                for key in ("infrastructure_error_rate", "admission_rejection_rate")
            )
        ]
        if failed:
            for axis, key in zip(axes, (throughput_key, latency_key), strict=True):
                axis.scatter(
                    [group["load"] for group in failed],
                    [group["metrics"][key]["mean"] for group in failed],
                    marker="x",
                    s=70,
                    color="red",
                    label="Rejection / infrastructure errors",
                    zorder=4,
                )
        axes[0].legend(frameon=False)
        fig.suptitle(
            "Independent run means +/- SD; single runs do not estimate variation",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "load-curves.png", dpi=160)
        plt.close(fig)
    if stage_names := report["spec"].get("stage_groups"):
        groups = [report["groups"][name] for name in stage_names]
        components = {
            "Repository hash before": "repository_digest_before_seconds",
            "Cache key before": "cache_key_before_seconds",
            "Model + tools + engine queue": "rollout_seconds",
            "Bounded context": "bounded_context_seconds",
            "Repository hash after": "repository_digest_after_seconds",
            "Cache key after": "cache_key_after_seconds",
            "MCP service queue": "service_queue_seconds",
        }
        fig, axis = plt.subplots(figsize=(11, 2.5 + len(groups) * 0.5))
        left = [0.0] * len(groups)
        for label, key in components.items():
            values = [
                group["metrics"][f"stage_metrics.{key}"]["mean"] for group in groups
            ]
            axis.barh(stage_names, values, left=left, label=label)
            left = [a + b for a, b in zip(left, values, strict=True)]
        other = [
            max(
                0,
                group["metrics"]["stage_metrics.service_total_seconds"]["mean"] - total,
            )
            for group, total in zip(groups, left, strict=True)
        ]
        axis.barh(stage_names, other, left=left, label="Other service work")
        axis.set(
            xlabel="Mean seconds per task, including failed tasks",
            title="Serial service stages; nested model/tool spans are not added twice",
        )
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False
        )
        fig.tight_layout()
        fig.savefig(output / "service-stages.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    for index, run in enumerate(report["steady"]):
        windows = run["complete_windows"]
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        minutes = [window["start_seconds"] / 60 for window in windows]
        axes[0].plot(
            minutes, [window["effective_tasks_per_minute"] for window in windows], "o-"
        )
        axes[0].set(
            ylabel="Effective tasks/min",
            title="One-minute completion windows; all failed tasks retained",
        )
        axes[1].plot(
            minutes, [window["latency_seconds"]["p95"] for window in windows], "o-"
        )
        axes[1].set(ylabel="Completion-cohort\nP95 latency (s)")
        axes[2].plot(
            minutes,
            [
                window["resources"]["vllm_waiting_requests"]["mean"]
                for window in windows
            ],
            "o-",
            label="Queued model requests",
        )
        axes[2].set(ylabel="Engine queue")
        for key, label in (
            ("gpu_utilization_percent", "GPU activity"),
            ("cpu_quota_utilization_percent", "Container CPU quota"),
        ):
            axes[3].plot(
                minutes,
                [window["resources"][key]["mean"] for window in windows],
                "o-",
                label=label,
            )
        axes[3].set(ylabel="Utilization (%)")
        axes[3].legend(frameon=False)
        for key, label in (
            ("memory_anon_bytes", "Host anonymous"),
            ("memory_file_bytes", "Host file cache"),
            ("gpu_memory_used_bytes", "GPU allocation"),
        ):
            axes[4].plot(
                minutes,
                [window["resources"][key]["mean"] / 2**30 for window in windows],
                "o-",
                label=label,
            )
        axes[4].set(xlabel="Elapsed minute", ylabel="GiB")
        axes[4].legend(frameon=False)
        for axis in axes:
            axis.axvspan(0, run["steady_start_seconds"] / 60, color="grey", alpha=0.15)
            axis.axvspan(
                run["submission_end_seconds"] / 60,
                run["total_seconds"] / 60,
                color="grey",
                alpha=0.15,
            )
            axis.grid(alpha=0.2)
        axes[4].set_xlabel("Elapsed minute; grey regions are ramp-up / drain")
        fig.tight_layout()
        fig.savefig(output / f"steady-{index + 1}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
