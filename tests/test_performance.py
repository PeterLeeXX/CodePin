import pytest

from src.performance import (
    TOOL_BEHAVIOR_MAX_FIELDS,
    analyze_token_trajectories,
    build_replay_workload,
    build_task_behavior_reference,
    evaluate_task_behavior,
    has_runtime_exception,
    histogram_quantile_delta,
    metric_delta,
    metric_value,
    parse_prometheus,
    source_manifest,
    summarize,
)


def test_server_runtime_failures_are_distinct_from_model_finish_failures():
    assert has_runtime_exception({"errors": ["AttributeError: missing state"]})
    assert has_runtime_exception({"errors": ["Exception: server failed"]})
    assert has_runtime_exception({"exception_type": "TimeoutError"})
    assert has_runtime_exception(
        {"errors": ["PydanticUndefinedAnnotation: missing tool schema"]}
    )
    assert not has_runtime_exception(
        {"errors": ["source mentions PydanticUndefinedAnnotation"]}
    )
    assert has_runtime_exception(
        {"errors": ["repository_or_deployment_changed_during_run"]}
    )
    assert not has_runtime_exception(
        {"errors": ["missing_or_multiple_finish", "tool_error"]}
    )


def behavior_record(
    task,
    *,
    effective=True,
    quality=1.0,
    file_f1=1.0,
    output_chars=100.0,
):
    return {
        "instance_id": task,
        "effective": effective,
        "quality": quality,
        "quality_metrics": {
            "file_f1": file_f1,
            "class_f1": 0.5,
            "function_f1": 0.25,
        },
        "metrics": dict.fromkeys(TOOL_BEHAVIOR_MAX_FIELDS, 0.0)
        | {"output_chars": output_chars},
    }


def test_task_behavior_gate_freezes_per_task_quality_and_tool_cost():
    reference = build_task_behavior_reference(
        [
            [behavior_record("task-a", quality=2.0, output_chars=90)],
            [behavior_record("task-a", quality=1.5, file_f1=0.8)],
        ]
    )
    accepted = evaluate_task_behavior(
        [behavior_record("task-a", quality=1.5, file_f1=0.8)], reference
    )
    assert accepted["accepted"]
    assert reference["per_task"]["task-a"]["records_per_reference_run"] == [1, 1]

    rejected = evaluate_task_behavior(
        [
            behavior_record(
                "task-a",
                effective=False,
                quality=1.4,
                file_f1=0.7,
                output_chars=101,
            ),
            behavior_record("unexpected"),
        ],
        reference,
    )
    fields = {row["field"] for row in rejected["regressions"]}
    assert not rejected["accepted"]
    assert {"effective_rate", "quality", "file_f1", "output_chars"} <= fields
    assert "unexpected_task" in fields


def test_task_behavior_gate_rejects_missing_or_mismatched_tasks():
    reference = build_task_behavior_reference([[behavior_record("task-a")]])
    result = evaluate_task_behavior([], reference)
    assert result["regressions"] == [{"task": "task-a", "field": "missing_task"}]
    with pytest.raises(ValueError, match="same tasks"):
        build_task_behavior_reference(
            [[behavior_record("task-a")], [behavior_record("task-b")]]
        )
    missing = behavior_record("task-a")
    del missing["metrics"]["tool_errors"]
    with pytest.raises(ValueError, match="missing or invalid tool measurement"):
        evaluate_task_behavior([missing], reference)
    invalid = behavior_record("task-a", quality=float("nan"))
    with pytest.raises(ValueError, match="non-finite quality measurement"):
        evaluate_task_behavior([invalid], reference)


def test_report_separates_completion_token_variation_from_tool_behavior():
    from scripts.report_performance import per_task

    first = behavior_record("task-a")
    first["latency_seconds"] = 1.0
    first["metrics"]["completion_tokens"] = 100
    second = {
        **first,
        "metrics": {**first["metrics"], "completion_tokens": 102},
    }
    before = per_task([first])["task-a"]
    after = per_task([first, second])["task-a"]
    assert before["behavior_values"] == after["behavior_values"]
    assert after["completion_tokens"]["min"] == 100
    assert after["completion_tokens"]["max"] == 102


def test_source_manifest_ignores_only_root_git_metadata(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/index").write_bytes(b"clone timestamp")
    (tmp_path / "code.py").write_bytes(b"code")
    first = source_manifest(tmp_path)
    (tmp_path / ".git/index").write_bytes(b"new timestamp")
    assert source_manifest(tmp_path) == first
    (tmp_path / "code.py").write_bytes(b"edit")
    assert source_manifest(tmp_path)["sha256"] != first["sha256"]
    (tmp_path / "nested/.git").mkdir(parents=True)
    assert "nested/.git" in {
        row["path"] for row in source_manifest(tmp_path)["entries"]
    }


def trajectory(instance_id, prompts, responses, *, broken=False):
    events = []
    for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        call_id = f"call-{index}"
        response_id = f"response-{index}"
        events.extend(
            [
                {
                    "kind": "ActionEvent",
                    "timestamp": f"2026-09-04T00:00:0{index}.000000",
                    "llm_response_id": response_id,
                    "tool_call_id": call_id,
                },
                {
                    "kind": "ObservationEvent",
                    "timestamp": f"2026-09-04T00:00:0{index}.125000",
                    "tool_call_id": call_id,
                },
                {
                    "kind": "TokenEvent",
                    "prompt_token_ids": prompt,
                    "response_token_ids": response,
                },
            ]
        )
    if broken:
        events[-1]["prompt_token_ids"][0] = 99
    return {"instance_id": instance_id, "status": "ok", "messages": events}


def test_append_only_analysis_uses_real_token_boundaries():
    first = trajectory("a", [[1, 2, 3], [1, 2, 3, 4, 9]], [[4], [5]])
    second = trajectory("b", [[1, 2, 8]], [[7]])
    report = analyze_token_trajectories([first, second], cache_block_size=2)

    assert report["all_tasks_strict_append_only"]
    second_turn = report["tasks"][0]["rounds"][1]
    assert second_turn["previous_prompt_is_prefix"]
    assert second_turn["previous_generation_is_prefix"]
    assert second_turn["appended_observation_tokens"] == 1
    # Token 4 was sampled, but has not necessarily been computed as an input.
    assert second_turn["block_aligned_reusable_tokens"] == 2
    assert report["cross_task_first_turn_pairs"] == [
        {
            "left": "a",
            "right": "b",
            "common_prefix_tokens": 2,
            "block_aligned_reusable_tokens": 2,
        }
    ]
    assert report["turns_per_task"]["mean"] == 1.5
    assert report["prompt_tokens_per_request"]["count"] == 3
    assert report["within_task_block_aligned_reusable_tokens"]["mean"] == 2


def test_append_only_analysis_detects_history_rewrite():
    broken = trajectory("broken", [[1, 2], [1, 2, 3]], [[3], [4]], broken=True)
    report = analyze_token_trajectories([broken], cache_block_size=2)
    assert not report["all_tasks_strict_append_only"]


def test_missing_token_evidence_cannot_pass_or_be_replayed():
    assert not analyze_token_trajectories([], cache_block_size=544)[
        "all_tasks_strict_append_only"
    ]
    empty = {"instance_id": "failed-before-inference", "messages": []}
    assert not analyze_token_trajectories([empty], cache_block_size=544)[
        "all_tasks_strict_append_only"
    ]
    with pytest.raises(ValueError, match="nonempty real"):
        build_replay_workload([empty])
    incomplete = trajectory("incomplete", [[]], [[]])
    assert not analyze_token_trajectories([incomplete], cache_block_size=544)[
        "all_tasks_strict_append_only"
    ]


def test_replay_workload_preserves_tokens_and_tool_delay():
    row = trajectory("task", [[1, 2]], [[3, 4]])
    workload = build_replay_workload([row])
    assert workload == [
        {
            "instance_id": "task",
            "rounds": [
                {
                    "turn": 0,
                    "prompt_token_ids": [1, 2],
                    "response_tokens": 2,
                    "tool_duration_seconds": 0.125,
                }
            ],
        }
    ]


def test_prometheus_counter_and_histogram_deltas():
    before = parse_prometheus(
        """
vllm:prefix_cache_hits_total{model_name="codepin"} 10
vllm:prompt_tokens_by_source_total{source="local_compute"} 4
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 6
vllm:time_to_first_token_seconds_bucket{le="0.1"} 1
vllm:time_to_first_token_seconds_bucket{le="0.5"} 2
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 2
"""
    )
    after = parse_prometheus(
        """
vllm:prefix_cache_hits_total{model_name="codepin"} 15
vllm:prompt_tokens_by_source_total{source="local_compute"} 7
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 11
vllm:time_to_first_token_seconds_bucket{le="0.1"} 2
vllm:time_to_first_token_seconds_bucket{le="0.5"} 5
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 6
"""
    )
    assert metric_delta(before, after, "vllm:prefix_cache_hits_total") == 5
    assert metric_value(after, "vllm:prompt_tokens_by_source_total") == 18
    assert (
        metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": "local_cache_hit"},
        )
        == 5
    )
    assert (
        histogram_quantile_delta(before, after, "vllm:time_to_first_token_seconds", 0.5)
        == 0.5
    )


def test_summary_reports_spread_and_percentiles():
    report = summarize([1, 2, 3])
    assert report["mean"] == 2
    assert report["stdev"] == 1
    assert report["p50"] == 2
    assert report["p95"] == pytest.approx(2.9)
    assert report["ci95_half_width"] > 0


def test_nsight_overlap_is_a_union_not_sum():
    from scripts.analyze_nsight import IntervalCoverage, merge_intervals

    intervals = [(3, 8), (1, 4), (10, 12), (12, 15), (5, 5)]
    assert merge_intervals(intervals) == [(1, 8), (10, 15)]
    coverage = IntervalCoverage(intervals)
    assert coverage.between(0, 20) == 12
    assert coverage.between(4, 11) == 5
    assert coverage.between(8, 10) == 0
    assert IntervalCoverage([]).between(0, 20) == 0


def test_cuda_details_separates_host_wait_and_process_gpu_overlap(tmp_path):
    import sqlite3

    from scripts.analyze_nsight import cuda_details

    path = tmp_path / "cuda.sqlite"
    pid, other_pid = 1 << 24, 2 << 24
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "INSERT INTO StringIds VALUES(1,'cudaEventSynchronize_v3020'),(2,'cudaLaunchKernel');"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,deviceId,contextId,streamId,correlationId);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid,deviceId,contextId,streamId,correlationId,bytes,copyKind,srcKind,dstKind);"
            "CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id,label);"
            "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'Host-to-Device');"
            "CREATE TABLE ENUM_CUDA_MEM_KIND(id,label);"
            "INSERT INTO ENUM_CUDA_MEM_KIND VALUES(1,'Pinned'),(2,'Device');"
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)",
            [
                (5, 80, pid + 7, 1, 1),
                (18, 35, pid + 7, 2, 2),
                (11, 13, other_pid + 4, 2, 2),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?)",
            [
                (20, 40, pid, 0, 1, 7, 2),
                (30, 50, pid, 0, 1, 9, 2),
                (10, 80, other_pid, 0, 2, 7, 2),
            ],
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (35, 45, pid, 0, 1, 8, 3, 512, 1, 1, 2),
        )
    report = cuda_details(path, 10, 70)
    wait = report["host_synchronization"][0]
    assert wait["host_wait_union_seconds"] == 60 / 1e9
    assert wait["overlapping_gpu_union_seconds"] == 30 / 1e9
    assert wait["without_traced_gpu_seconds"] == 30 / 1e9
    copy = report["memory_copies"][0]
    assert copy["source_memory"] == "Pinned"
    assert copy["bytes"] == 512
    assert copy["overlapping_compute_union_seconds"] == 10 / 1e9
    assert len(report["streams"]) == 4
    assert report["runtime_correlations_with_gpu_activity"] == 2
    # Two kernels sharing one launch count once; another process may reuse its ID.
    delay = report["api_return_to_first_gpu_seconds"][0]
    assert delay["count"] == 2
    assert delay["mean"] == pytest.approx(-9 / 1e9)


def test_three_repeat_interval_uses_student_t():
    assert summarize([1, 2, 3])["ci95_half_width"] == pytest.approx(2.484138, rel=1e-6)
    assert summarize([1])["ci95_half_width"] is None


@pytest.mark.parametrize("has_kernel,expected", [(True, 35), (False, 30)])
def test_nsight_gpu_graphs_are_not_reported_as_idle(tmp_path, has_kernel, expected):
    import sqlite3

    from scripts.analyze_nsight import analyze

    path = tmp_path / "graph.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, "
            "textId INTEGER, globalTid INTEGER);"
            "INSERT INTO NVTX_EVENTS VALUES(0,100,'codepin.benchmark',NULL,1);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, "
            "globalPid INTEGER, shortName INTEGER, demangledName INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, "
            "nameId INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER, "
            "end INTEGER, globalPid INTEGER);"
            "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(15,45,2);"
        )
        if has_kernel:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,20,2,1,1)"
            )
    report, _, intervals = analyze(path)
    assert report["gpu_busy_union_seconds"] == expected / 1e9
    assert report["gpu_global_pids"] == [2]
    assert report["gpu_operation_counts"]["graph_trace"] == 1
    assert report["graph_granularity"]
    assert len(intervals) == 1


def test_tool_to_gpu_attribution_requires_serial_tasks():
    from scripts.analyze_nsight import single_task_handoffs

    ranges = [
        (0, 100, "codepin.task|first", 1),
        (1, 20, "codepin.step|conversation|events=0", 2),
        (16, 18, "codepin.tool.read_file", 2),
        (24, 90, "codepin.step|conversation|events=3", 2),
    ]
    result = single_task_handoffs(ranges, [(5, 10), (30, 80)], 0)
    assert result["records"][0]["tool_to_next_gpu_seconds"] == 12 / 1e9
    assert result["records"][0]["tool_to_next_step_seconds"] == 6 / 1e9
    with pytest.raises(ValueError, match="nonoverlapping"):
        single_task_handoffs(
            ranges + [(25, 80, "codepin.task|second", 3)], [(30, 80)], 0
        )


def test_steady_window_keeps_late_failed_results_in_latency():
    from scripts.report_performance import continuous_window

    rows = [
        {"submitted_offset_seconds": 2, "latency_seconds": 10, "effective": True},
        {"submitted_offset_seconds": 11, "latency_seconds": 3, "effective": False},
        {"submitted_offset_seconds": 19, "latency_seconds": 50, "effective": False},
        {"submitted_offset_seconds": 24, "latency_seconds": 1, "effective": True},
    ]
    result = continuous_window(
        {"config": {"continuous": True, "minimum_duration": 20}},
        rows,
        10,
        [
            {"elapsed_seconds": 2, "gpu_utilization_percent": 5},
            {"elapsed_seconds": 12, "gpu_utilization_percent": 90},
            {"elapsed_seconds": 24, "gpu_utilization_percent": 0},
        ],
    )
    assert result["steady_terminal_count"] == 2
    assert result["steady_effective_tasks_per_minute"] == 6
    assert result["steady_submitted_count"] == 2
    assert result["steady_effective_rate"] == 0
    assert result["steady_latency_seconds.p95"] == pytest.approx(47.65)
    assert result["steady_resources.gpu_utilization_percent"] == 90
