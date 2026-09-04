"""Summarize a real Nsight SQLite export without double-counting GPU overlap."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from src.performance import summarize
from src.profiling import issue_trace_id


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class IntervalCoverage:
    def __init__(self, intervals):
        self.intervals = merge_intervals(intervals)
        self.starts = [start for start, _ in self.intervals]
        self.accumulated = [0]
        for start, end in self.intervals:
            self.accumulated.append(self.accumulated[-1] + end - start)

    def before(self, point):
        index = bisect.bisect_left(self.starts, point) - 1
        if index < 0:
            return 0
        start, end = self.intervals[index]
        return self.accumulated[index] + min(point, end) - start

    def between(self, start, end):
        return self.before(end) - self.before(start)


def cuda_details(path: Path, begin: int, end: int) -> dict:
    """Correlate native CUDA API waits, streams and copies in an explicit window."""
    if begin >= end:
        raise ValueError("CUDA analysis window must have positive duration")
    streams, compute, copies, waits = (defaultdict(list) for _ in range(4))
    stream_counts, copy_bytes = defaultdict(int), defaultdict(int)
    api_calls, first_gpu = {}, {}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for start, stop, tid, correlation, name in connection.execute(
            "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "WHERE r.start < ? AND r.end > ?",
            (end, begin),
        ):
            pid = tid & ~0xFFFFFF
            base_name = name.rsplit("_v", 1)[0]
            if base_name in {
                "cudaEventSynchronize",
                "cudaStreamSynchronize",
                "cudaDeviceSynchronize",
                "cuEventSynchronize",
                "cuStreamSynchronize",
                "cuCtxSynchronize",
            }:
                waits[pid, name].append((max(start, begin), min(stop, end)))
            if correlation is not None:
                key = pid, correlation
                if key in api_calls:
                    raise ValueError(f"ambiguous runtime correlation in capture: {key}")
                api_calls[key] = (start, stop, name, tid)
        for kind in ("KERNEL", "MEMCPY", "MEMSET", "GRAPH_TRACE"):
            table = "CUPTI_ACTIVITY_KIND_" + kind
            if table not in tables:
                continue
            extra = ",bytes,copyKind,srcKind,dstKind" if kind == "MEMCPY" else ""
            for row in connection.execute(
                f"SELECT start,end,globalPid,deviceId,contextId,streamId,correlationId{extra} "
                f"FROM {table} WHERE start < ? AND end > ?",
                (end, begin),
            ):
                start, stop, pid, device, context, stream, correlation, *tail = row
                clipped = max(start, begin), min(stop, end)
                key = pid, device, context, stream
                streams[key].append(clipped)
                stream_counts[key, kind] += 1
                if kind in {"KERNEL", "GRAPH_TRACE"}:
                    compute[pid].append(clipped)
                if kind == "MEMCPY":
                    size, direction, source, destination = tail
                    copy_key = pid, direction, source, destination
                    copies[copy_key].append(clipped)
                    copy_bytes[copy_key] += size
                if correlation is not None and (pid, correlation) in api_calls:
                    key = pid, correlation
                    if key not in first_gpu or start < first_gpu[key][0]:
                        first_gpu[key] = start, kind
        labels = {}
        for table in ("ENUM_CUDA_MEMCPY_OPER", "ENUM_CUDA_MEM_KIND"):
            labels[table] = (
                dict(connection.execute(f"SELECT id,label FROM {table}"))
                if table in tables
                else {}
            )

    gpu_by_pid = defaultdict(list)
    for (pid, *_), intervals in streams.items():
        gpu_by_pid[pid].extend(intervals)
    gpu_by_pid = {pid: IntervalCoverage(values) for pid, values in gpu_by_pid.items()}
    compute = {pid: IntervalCoverage(values) for pid, values in compute.items()}
    sync_rows = []
    for (pid, name), values in sorted(waits.items()):
        coverage = gpu_by_pid.get(pid, IntervalCoverage([]))
        union = merge_intervals(values)
        union_ns = sum(stop - start for start, stop in union)
        overlap_ns = sum(coverage.between(start, stop) for start, stop in union)
        sync_rows.append(
            {
                "global_pid": pid,
                "name": name,
                "count": len(values),
                "host_call_seconds": summarize(
                    (stop - start) / 1e9 for start, stop in values
                ),
                "host_wait_union_seconds": union_ns / 1e9,
                "overlapping_gpu_union_seconds": overlap_ns / 1e9,
                "without_traced_gpu_seconds": (union_ns - overlap_ns) / 1e9,
            }
        )
    copy_rows = []
    for key, values in sorted(copies.items()):
        pid, direction, source, destination = key
        union = merge_intervals(values)
        coverage = compute.get(pid, IntervalCoverage([]))
        copy_rows.append(
            {
                "global_pid": pid,
                "direction": labels["ENUM_CUDA_MEMCPY_OPER"].get(
                    direction, str(direction)
                ),
                "source_memory": labels["ENUM_CUDA_MEM_KIND"].get(source, str(source)),
                "destination_memory": labels["ENUM_CUDA_MEM_KIND"].get(
                    destination, str(destination)
                ),
                "count": len(values),
                "bytes": copy_bytes[key],
                "copy_resource_seconds": sum(stop - start for start, stop in values)
                / 1e9,
                "copy_union_seconds": sum(stop - start for start, stop in union) / 1e9,
                "overlapping_compute_union_seconds": sum(
                    coverage.between(start, stop) for start, stop in union
                )
                / 1e9,
            }
        )
    delays = defaultdict(list)
    for key, (start, kind) in first_gpu.items():
        _, stop, name, _ = api_calls[key]
        delays[name, kind].append((start - stop) / 1e9)
    return {
        "window_seconds": (end - begin) / 1e9,
        "available_activity_tables": sorted(
            t for t in tables if t.startswith("CUPTI_ACTIVITY_KIND_")
        ),
        "streams": [
            {
                "global_pid": key[0],
                "device": key[1],
                "context": key[2],
                "stream": key[3],
                "gpu_union_seconds": IntervalCoverage(values).between(begin, end) / 1e9,
                "operation_counts": {
                    kind: count
                    for (stream_key, kind), count in stream_counts.items()
                    if stream_key == key
                },
            }
            for key, values in sorted(streams.items())
        ],
        "host_synchronization": sync_rows,
        "memory_copies": copy_rows,
        "runtime_correlations": len(api_calls),
        "runtime_correlations_with_gpu_activity": len(first_gpu),
        "api_return_to_first_gpu_seconds": [
            {"api": name, "first_operation": kind, **summarize(values)}
            for (name, kind), values in sorted(delays.items())
        ],
        "interpretation": "Intervals are clipped to the selected window; copy bytes count each intersecting operation in full. Synchronization is host API time and its temporal overlap with this process's GPU work, not avoidable waiting or causal attribution. Stream and copy unions must not be added to obtain a task critical path. Graph intervals contain unobserved internals, so compute overlap is an upper bound with graph-level tracing. Runtime correlation uses process plus correlation ID; grouped GPU operations count once per API invocation. Negative API-return-to-GPU delay means GPU execution started before the host API returned. This delay is not vLLM request queue time.",
    }


def analyze(path: Path) -> tuple[dict, list, list]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        ranges = list(
            connection.execute(
                "SELECT n.start, n.end, COALESCE(n.text,s.value), n.globalTid "
                "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                "WHERE n.end IS NOT NULL ORDER BY n.start"
            )
        )
        capture = [
            (start, end)
            for start, end, name, _ in ranges
            if name == "codepin.benchmark"
        ]
        if len(capture) != 1:
            raise ValueError("expected exactly one codepin.benchmark capture range")
        begin, end = capture[0]
        kernels = list(
            connection.execute(
                "SELECT start,end,globalPid,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start >= ? AND end <= ? ORDER BY start",
                (begin, end),
            )
        )
        kernel_coverage = IntervalCoverage((row[0], row[1]) for row in kernels)
        gpu_operations = [(row[0], row[1]) for row in kernels]
        gpu_pids = {row[2] for row in kernels}
        operation_counts = {"kernel": len(kernels)}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for name in ("MEMCPY", "MEMSET", "GRAPH_TRACE"):
            table = "CUPTI_ACTIVITY_KIND_" + name
            if table in tables:
                intervals = list(
                    connection.execute(
                        f"SELECT start,end FROM {table} WHERE start >= ? AND end <= ?",
                        (begin, end),
                    )
                )
                operation_counts[name.lower()] = len(intervals)
                gpu_operations.extend(intervals)
                if name == "GRAPH_TRACE":
                    gpu_pids.update(
                        row[0]
                        for row in connection.execute(
                            f"SELECT DISTINCT globalPid FROM {table} "
                            "WHERE start >= ? AND end <= ?",
                            (begin, end),
                        )
                    )
        if not gpu_operations:
            raise ValueError("trace has no GPU activity in the benchmark range")
        gpu = IntervalCoverage(gpu_operations)
        gaps = [
            (left[1], right[0])
            for left, right in zip(gpu.intervals, gpu.intervals[1:])
            if right[0] > left[1]
        ]
        gaps = [(begin, gpu.intervals[0][0]), *gaps, (gpu.intervals[-1][1], end)]
        gaps = [(start, stop) for start, stop in gaps if stop > start]
        digest_coverage = IntervalCoverage(
            (start, stop)
            for start, stop, name, _ in ranges
            if name
            in {"codepin.repository_digest_before", "codepin.repository_digest_after"}
        )
        stage_values = defaultdict(list)
        stage_gpu = defaultdict(float)
        for start, stop, name, _ in ranges:
            if start < begin or stop > end or not name:
                continue
            if name.startswith(("codepin.", "gpu_model_runner:", "schedule:")):
                group = name.split("|")[0]
                stage_values[group].append((stop - start) / 1e9)
                stage_gpu[group] += gpu.between(start, stop) / 1e9
        top_kernels = list(
            connection.execute(
                "SELECT s.value,COUNT(*),SUM(k.end-k.start)/1e9 "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                "WHERE k.start >= ? AND k.end <= ? GROUP BY k.demangledName "
                "ORDER BY SUM(k.end-k.start) DESC LIMIT 20",
                (begin, end),
            )
        )
        runtime = list(
            connection.execute(
                "SELECT s.value,COUNT(*),SUM(r.end-r.start)/1e9 "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
                "WHERE r.start >= ? AND r.end <= ? GROUP BY r.nameId "
                "ORDER BY SUM(r.end-r.start) DESC LIMIT 15",
                (begin, end),
            )
        )
        osrt_pids = (
            [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT (globalTid >> 24) & 16777215 FROM OSRT_API"
                )
            ]
            if "OSRT_API" in tables
            else []
        )
    report = {
        "sqlite": str(path),
        "capture_start_ns": begin,
        "capture_end_ns": end,
        "capture_seconds": (end - begin) / 1e9,
        "kernel_count": len(kernels),
        "gpu_operation_counts": operation_counts,
        "gpu_global_pids": sorted(pid for pid in gpu_pids if pid is not None),
        "nvtx_process_ids": sorted(
            {(row[3] >> 24) & 0xFFFFFF for row in ranges if row[3]}
        ),
        "osrt_process_ids": sorted(osrt_pids),
        "kernel_busy_union_seconds": kernel_coverage.between(begin, end) / 1e9,
        "gpu_busy_union_seconds": gpu.between(begin, end) / 1e9,
        "gpu_busy_fraction": gpu.between(begin, end) / (end - begin),
        "kernel_resource_seconds": sum(stop - start for start, stop, *_ in kernels)
        / 1e9,
        "gpu_idle_gap_seconds": summarize((stop - start) / 1e9 for start, stop in gaps),
        "gaps_over_1ms_seconds": sum(
            stop - start for start, stop in gaps if stop - start > 1e6
        )
        / 1e9,
        "gaps_over_10ms_seconds": sum(
            stop - start for start, stop in gaps if stop - start > 1e7
        )
        / 1e9,
        "long_gap_digest_overlap_union_seconds": sum(
            digest_coverage.between(start, stop)
            for start, stop in gaps
            if stop - start > 1e7
        )
        / 1e9,
        "longest_gaps": [
            {
                "start_seconds": (start - begin) / 1e9,
                "duration_seconds": (stop - start) / 1e9,
                "digest_overlap_seconds": digest_coverage.between(start, stop) / 1e9,
            }
            for start, stop in sorted(
                gaps, key=lambda pair: pair[1] - pair[0], reverse=True
            )[:20]
        ],
        "nvtx_stages": {
            name: summarize(values) for name, values in stage_values.items()
        },
        "stage_gpu_overlap_resource_seconds": dict(stage_gpu),
        "top_kernels": [
            {"name": name, "count": count, "resource_seconds": seconds}
            for name, count, seconds in top_kernels
        ],
        "top_cuda_apis": [
            {"name": name, "count": count, "resource_seconds": seconds}
            for name, count, seconds in runtime
        ],
        "graph_granularity": bool(operation_counts.get("graph_trace")),
        "interpretation": "GPU busy is the union of traced kernel, memcpy, memset and GPU graph execution intervals, not SM utilization. Graph-level intervals can contain unobserved internal gaps; their union is an upper bound on device activity and cannot be compared directly with node-level busy fractions. Kernel counts and sums omit graph internals when graph-level tracing is used. Per-stage and kernel sums are cumulative resource time, not an additive task critical path. Overlap is temporal correlation, not proof of causality.",
    }
    return report, ranges, gpu.intervals


def correlate_steps(ranges: list, trajectories: Path, origin: int) -> list[dict]:
    """Join NVTX event indices to native OpenHands response/tool IDs."""
    conversations = {}
    for path in trajectories.glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("conversation_id"):
            conversations[record["conversation_id"]] = record
    steps = defaultdict(list)
    for start, end, name, tid in ranges:
        if name and name.startswith("codepin.step|"):
            _, conversation_id, events = name.split("|")
            steps[conversation_id].append(
                (int(events.removeprefix("events=")), start, end, tid)
            )
    output = []
    for conversation_id, entries in steps.items():
        if conversation_id not in conversations:
            raise ValueError(f"missing native trajectory for {conversation_id}")
        record = conversations[conversation_id]
        events = record["messages"]
        entries.sort()
        for turn, (offset, start, end, tid) in enumerate(entries):
            stop = entries[turn + 1][0] if turn + 1 < len(entries) else len(events)
            segment = events[offset:stop]
            output.append(
                {
                    "conversation_id": conversation_id,
                    "issue_id": record.get("issue_id")
                    or issue_trace_id(record["instance"]["problem_statement"]),
                    "turn": turn,
                    "event_start": offset,
                    "event_end": stop,
                    "start_seconds": (start - origin) / 1e9,
                    "end_seconds": (end - origin) / 1e9,
                    "global_tid": tid,
                    "llm_response_ids": sorted(
                        {
                            event["llm_response_id"]
                            for event in segment
                            if event.get("llm_response_id")
                        }
                    ),
                    "tool_call_ids": sorted(
                        {
                            event["tool_call_id"]
                            for event in segment
                            if event.get("tool_call_id")
                        }
                    ),
                    "tools": [
                        event["tool_name"]
                        for event in segment
                        if event.get("kind") == "ActionEvent"
                    ],
                }
            )
    return sorted(output, key=lambda record: record["start_seconds"])


def single_task_handoffs(ranges: list, gpu_intervals: list, origin: int) -> dict:
    """Measure tool-to-next-GPU gaps only when task ranges prove serial execution."""
    tasks = sorted(
        (start, end)
        for start, end, name, _ in ranges
        if name and name.startswith("codepin.task|")
    )
    if not tasks or any(a[1] > b[0] for a, b in pairwise(tasks)):
        raise ValueError(
            "single-task handoff attribution requires nonoverlapping tasks"
        )
    grouped = defaultdict(list)
    tools = []
    for start, end, name, tid in ranges:
        if name and name.startswith("codepin.step|"):
            grouped[name.split("|")[1]].append((start, end, tid))
        elif name and name.startswith("codepin.tool."):
            tools.append((start, end, name, tid))
    starts = [start for start, _ in gpu_intervals]
    rows = []
    for conversation_id, steps in grouped.items():
        steps.sort()
        for previous, following in pairwise(steps):
            completed = [
                tool
                for tool in tools
                if tool[3] == previous[2]
                and previous[0] <= tool[0] < tool[1] <= previous[1]
            ]
            if not completed:
                continue
            tool = max(completed, key=lambda item: item[1])
            index = bisect.bisect_left(starts, following[0])
            if index == len(starts) or starts[index] >= following[1]:
                raise ValueError("next Agent step has no traced GPU work")
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "tool": tool[2],
                    "tool_end_seconds": (tool[1] - origin) / 1e9,
                    "next_step_start_seconds": (following[0] - origin) / 1e9,
                    "next_gpu_start_seconds": (starts[index] - origin) / 1e9,
                    "tool_to_next_step_seconds": (following[0] - tool[1]) / 1e9,
                    "tool_to_next_gpu_seconds": (starts[index] - tool[1]) / 1e9,
                }
            )
    if not rows:
        raise ValueError("no consecutive real tool/model steps to attribute")
    return {
        "records": rows,
        "tool_to_next_step_seconds": summarize(
            row["tool_to_next_step_seconds"] for row in rows
        ),
        "tool_to_next_gpu_seconds": summarize(
            row["tool_to_next_gpu_seconds"] for row in rows
        ),
        "interpretation": "Serial-task diagnostic only: the next step's first traced GPU operation includes request construction, HTTP, native rendering and scheduling. It is not an estimate for overlapping multi-task execution.",
    }


def timeline(report, ranges, gpu_intervals, output: Path, max_lanes: int = 12):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    origin = report["capture_start_ns"]
    window_start = max(0, report["capture_seconds"] / 2 - 2)
    window_end = window_start + 4
    begin, end = origin + window_start * 1e9, origin + window_end * 1e9
    colors = {
        "codepin.repository_digest_before": "#ad5a00",
        "codepin.repository_digest_after": "#e5a449",
        "codepin.conversation_run": "#2878b5",
        "codepin.cache_key_before": "#9e9e9e",
        "codepin.cache_key_after": "#9e9e9e",
        "codepin.bounded_context": "#499f68",
    }
    lanes = sorted(
        {
            tid
            for start, stop, name, tid in ranges
            if name in colors and start < end and stop > begin
        }
    )
    total_lanes = len(lanes)
    lanes = lanes[:max_lanes]
    fig, axis = plt.subplots(figsize=(12, max(4, 2.5 + len(lanes) * 0.5)))

    def clipped(intervals):
        return [
            (
                (max(start, begin) - origin) / 1e9,
                (min(stop, end) - max(start, begin)) / 1e9,
            )
            for start, stop in intervals
            if start < end and stop > begin
        ]

    coverage = IntervalCoverage(gpu_intervals)
    bin_width = (end - begin) / 2000
    activity = [
        coverage.between(begin + i * bin_width, begin + (i + 1) * bin_width) / bin_width
        for i in range(2000)
    ]
    axis.imshow(
        [activity],
        extent=(window_start, window_end, 0, 0.7),
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="auto",
        origin="lower",
    )
    for index, tid in enumerate(lanes, 1):
        for name, color in colors.items():
            axis.broken_barh(
                clipped(
                    (start, stop)
                    for start, stop, label, thread in ranges
                    if label == name and thread == tid
                ),
                (index, 0.7),
                facecolors=color,
            )
    axis.set(
        yticks=[0.35] + [i + 0.35 for i in range(1, len(lanes) + 1)],
        yticklabels=[
            "GPU execution envelope / 2 ms"
            if report["graph_granularity"]
            else "GPU active fraction / 2 ms"
        ]
        + [f"Agent worker {i + 1}" for i in range(len(lanes))],
        xlim=(window_start, window_end),
        xlabel="Seconds since benchmark capture started",
        title=(
            "Nsight trace: overlapping GPU work and CodePin stages\n"
            f"Showing {len(lanes)} of {total_lanes} active Agent threads; full trace retained"
        ),
    )
    fig.legend(
        handles=[
            Patch(color=color, label=name.removeprefix("codepin."))
            for name, color in list(colors.items())[:3]
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--benchmark-records", type=Path)
    parser.add_argument("--timeline-max-lanes", type=int, default=12)
    parser.add_argument("--single-task-handoff", action="store_true")
    parser.add_argument("--cuda-details", action="store_true")
    parser.add_argument(
        "--steady-window", nargs=2, type=float, metavar=("START", "END")
    )
    args = parser.parse_args()
    if args.timeline_max_lanes < 1:
        parser.error("timeline-max-lanes must be positive")
    if args.output.exists() or (args.timeline and args.timeline.exists()):
        raise FileExistsError("analysis output already exists")
    report, ranges, intervals = analyze(args.sqlite)
    detail_begin, detail_end = report["capture_start_ns"], report["capture_end_ns"]
    if args.steady_window:
        start, end = args.steady_window
        if not 0 <= start < end <= report["capture_seconds"]:
            parser.error("steady-window must lie inside the benchmark capture")
        coverage = IntervalCoverage(intervals)
        origin = report["capture_start_ns"]
        busy = coverage.between(origin + start * 1e9, origin + end * 1e9) / 1e9
        report["steady_window"] = {
            "start_seconds": start,
            "end_seconds": end,
            "gpu_busy_union_seconds": busy,
            "gpu_busy_fraction": busy / (end - start),
        }
        detail_begin, detail_end = int(origin + start * 1e9), int(origin + end * 1e9)
    if args.cuda_details:
        report["cuda_details"] = cuda_details(args.sqlite, detail_begin, detail_end)
    if args.trajectories:
        report["agent_steps"] = correlate_steps(
            ranges, args.trajectories, report["capture_start_ns"]
        )
    if args.benchmark_records:
        if not args.trajectories:
            parser.error("benchmark-records requires trajectories")
        records = [
            json.loads(line)
            for line in args.benchmark_records.read_text().splitlines()
            if line.strip()
        ]
        executions = {
            row["execution_id"]: row for row in records if row.get("execution_id")
        }
        if len(executions) != sum(bool(row.get("execution_id")) for row in records):
            raise ValueError(
                "benchmark execution IDs must be unique; result cache must be off"
            )
        for step in report["agent_steps"]:
            record = executions[step["conversation_id"]]
            step.update(
                task_instance_id=record["instance_id"],
                task_cycle=record["cycle"],
                task_position=record["position"],
            )
    if args.single_task_handoff:
        report["single_task_handoffs"] = single_task_handoffs(
            ranges, intervals, report["capture_start_ns"]
        )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.timeline:
        timeline(report, ranges, intervals, args.timeline, args.timeline_max_lanes)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
