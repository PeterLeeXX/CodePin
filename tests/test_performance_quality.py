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


def context_outcome(quality=0.0, output_chars=69, tool_errors=0):
    row = outcome(quality=quality, tool_errors=tool_errors)
    row.update(status="ok" if tool_errors == 0 else "error", errors=[])
    row["metrics"]["output_chars"] = output_chars
    return row


def test_improved_localization_context_is_an_explicit_cost_tradeoff():
    baseline = [context_outcome(tool_errors=4)] * 100
    reference = freeze_reference([baseline] * 3, allow_quality_improving_context=True)
    original = copy.deepcopy(reference)
    candidate = baseline[:99] + [context_outcome(quality=0.2, output_chars=9621)]
    result = compare_records(candidate, reference)
    assert result["accepted"]
    assert {row["field"] for row in result["context_cost_tradeoffs"]} == {
        "output_chars",
        "mean_output_chars",
    }
    assert result["run_means"]["task"]["tool_cost"]["output_chars"] > 69
    assert result["run_means"]["task"]["tool_cost"]["tool_errors"] < 4
    assert result["all_outcome_counts"]["task"]["effective"] == 0
    assert reference == original


def test_historical_strict_references_keep_their_original_verdict():
    baseline = [[context_outcome()]] * 3
    reference = freeze_reference(baseline)
    assert "context_cost_policy" not in reference
    assert not compare_records(
        [context_outcome(quality=0.2, output_chars=9621)], reference
    )["accepted"]


@pytest.mark.parametrize(
    "kind", ["unchanged", "below_best", "broken_finish", "tool_error"]
)
def test_context_growth_requires_valid_localization_better_than_baseline_best(kind):
    baseline = [context_outcome()]
    if kind == "below_best":
        baseline.append(context_outcome(quality=1.0))
    reference = freeze_reference([baseline] * 3, allow_quality_improving_context=True)
    candidate = context_outcome(quality=0.2, output_chars=9621)
    if kind == "unchanged":
        candidate = context_outcome(output_chars=9621)
    elif kind == "broken_finish":
        candidate.update(status="error", errors=["missing_or_multiple_finish"])
    elif kind == "tool_error":
        candidate["metrics"]["tool_errors"] = 1
    result = compare_records(baseline + [candidate], reference)
    assert not result["accepted"]
    assert not result["context_cost_tradeoffs"]


def test_better_quality_cannot_hide_other_ordinary_outcome_cost_growth():
    baseline = [context_outcome()]
    reference = freeze_reference([baseline] * 3, allow_quality_improving_context=True)
    result = compare_records(
        [
            context_outcome(quality=0.2, output_chars=9621),
            context_outcome(output_chars=300),
        ],
        reference,
    )
    assert not result["accepted"]
    assert not result["context_cost_tradeoffs"]


def test_context_tradeoff_does_not_bypass_a_localization_level_regression():
    baseline = [context_outcome(quality=0.5)]
    reference = freeze_reference([baseline] * 3, allow_quality_improving_context=True)
    candidate = context_outcome(quality=0.7, output_chars=9621)
    candidate["quality_metrics"]["function_f1"] = 0.4
    result = compare_records([candidate], reference)
    assert not result["accepted"]
    assert not result["context_cost_tradeoffs"]


def test_context_tradeoff_never_excuses_repeated_search_or_extra_rounds():
    baseline = [context_outcome()]
    reference = freeze_reference([baseline] * 3, allow_quality_improving_context=True)
    candidate = context_outcome(quality=0.2, output_chars=9621)
    candidate["metrics"].update(repeated_searches=1, num_turns=1)
    result = compare_records([candidate], reference)
    assert not result["accepted"]
    assert result["context_cost_tradeoffs"]
    assert "repeated_searches" in {row["field"] for row in result["regressions"]}
