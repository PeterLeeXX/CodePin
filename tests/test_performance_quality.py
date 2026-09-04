import copy

import pytest

from scripts.compare_performance_quality import (
    compare_records,
    freeze_reference,
)
from scripts.report_performance import per_task
from src.performance import TOOL_BEHAVIOR_MAX_FIELDS


def outcome(task="task", quality=1.0, tool_errors=0):
    return {
        "instance_id": task,
        "effective": False,
        "quality": quality,
        "quality_metrics": dict.fromkeys(
            ("file_f1", "class_f1", "function_f1"), quality
        ),
        "metrics": dict.fromkeys(TOOL_BEHAVIOR_MAX_FIELDS, 0)
        | {"tool_errors": tool_errors},
    }


def test_tool_error_frequency_cannot_hide_behind_unchanged_extrema():
    baseline = [outcome()] * 99 + [outcome(tool_errors=1)]
    reference = freeze_reference([baseline, baseline, baseline])
    original = copy.deepcopy(reference)
    result = compare_records([outcome()] * 95 + [outcome(tool_errors=1)] * 5, reference)
    assert not result["accepted"]
    assert {row["field"] for row in result["regressions"]} == {"mean_tool_errors"}
    assert reference == original


def test_quality_frequency_is_checked_for_each_task_without_pooling():
    baseline = [outcome(quality=1)] * 9 + [outcome(quality=0)]
    baseline += [outcome("easy")] * 10
    reference = freeze_reference([baseline, baseline, baseline])
    candidate = [outcome(quality=1)] * 8 + [outcome(quality=0)] * 2
    candidate += [outcome("easy")] * 1000
    result = compare_records(candidate, reference)
    assert not result["accepted"]
    assert {row["task"] for row in result["regressions"]} == {"task"}
    assert "mean_function_f1" in {row["field"] for row in result["regressions"]}


def test_task_volume_alone_does_not_change_empirical_envelope():
    baseline = [outcome()] * 99 + [outcome(tool_errors=1)]
    reference = freeze_reference([baseline, baseline * 2, baseline * 3])
    assert compare_records(baseline * 10, reference)["accepted"]


def test_baseline_bounds_do_not_expand_to_confidence_interval():
    reference = freeze_reference(
        [
            [outcome(quality=0.9)],
            [outcome(quality=1.0)],
            [outcome(quality=0.95)],
        ]
    )
    assert not compare_records([outcome(quality=0.89)], reference)["accepted"]
    with pytest.raises(ValueError, match="three independent"):
        freeze_reference([[outcome()], [outcome()]])
    with pytest.raises(ValueError, match="non-finite"):
        compare_records([outcome(quality=float("nan"))], reference)


def test_runtime_failures_stay_counted_and_cannot_lower_model_quality_bounds():
    error = {
        "instance_id": "task",
        "effective": False,
        "status": "error",
        "exception_type": "RuntimeError",
        "errors": ["duplicate tool registration"],
    }
    baseline = [outcome(), error]
    reference = freeze_reference([baseline, baseline, baseline])
    assert reference["all_baseline_outcome_counts"][0]["task"]["submitted"] == 2
    assert (
        reference["all_baseline_outcome_counts"][0]["task"]["infrastructure_failures"]
        == 1
    )
    assert (
        reference["strict_reference"]["per_task"]["task"]["minimum_quality"]["quality"]
        == 1
    )
    assert not compare_records([outcome(quality=0.99)], reference)["accepted"]
    result = compare_records([outcome(), error], reference)
    assert not result["accepted"]
    assert result["all_outcome_counts"]["task"]["submitted"] == 2
    assert {row["field"] for row in result["regressions"]} == {
        "infrastructure_failures"
    }
    with pytest.raises(ValueError, match="measured trajectories"):
        freeze_reference([[error], [error], [error]])


def test_missing_measurements_without_runtime_failure_are_rejected():
    reference = freeze_reference([[outcome()], [outcome()], [outcome()]])
    incomplete = outcome()
    del incomplete["quality"]
    with pytest.raises(ValueError, match="missing or invalid quality"):
        compare_records([incomplete], reference)


def test_task_f1_report_keeps_startup_failure_in_denominator():
    success = outcome() | {"latency_seconds": 1.0}
    failed = {
        "instance_id": "task",
        "effective": False,
        "latency_seconds": 0.1,
        "exception_type": "RuntimeError",
        "errors": ["tool schema startup failed"],
    }
    report = per_task([success, failed])["task"]
    assert report["count"] == 2
    for score in report["quality_metrics"].values():
        assert score["mean"] == 0.5
        assert score["count"] == 2
