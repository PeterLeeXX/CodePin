import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data_pipeline import clean_tasks, export_data, validate_trajectory
from src.evaluate import downstream_summary, evaluate, load_downstream
from src.trajectory import score_trajectory, tool_metrics, validate_events


def task():
    return {
        "instance_id": "case1",
        "repo": "example/project",
        "base_commit": "a" * 40,
        "problem_statement": "Repair wrong addition",
        "use_patch": False,
        "file_changes": [
            {
                "file": "a.py",
                "changes": {
                    "edited_modules": ["a.py:add"],
                    "edited_entities": ["a.py:add"],
                    "added_modules": [],
                    "added_entities": [],
                },
            }
        ],
    }


def trajectory():
    locations = [{"file": "a.py", "function_name": "add"}]
    events = []
    chats = [{"role": "user", "content": "Repair wrong addition"}]
    for index, (name, args, obs) in enumerate(
        [
            (
                "grep",
                {"pattern": "add"},
                {"content": [{"text": "a.py:1:def add(a,b):"}]},
            ),
            (
                "read_file",
                {"path": "a.py"},
                {
                    "path": "a.py",
                    "start_line": 1,
                    "end_line": 2,
                    "content": [{"text": "def add(a,b):\n return a-b"}],
                },
            ),
            (
                "localization_finish",
                {"locations": locations},
                {"content": [{"text": json.dumps(locations)}]},
            ),
        ]
    ):
        call_id = str(index)
        events.extend(
            [
                {
                    "kind": "ActionEvent",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "llm_response_id": call_id,
                    "action": args,
                },
                {
                    "kind": "ObservationEvent",
                    "tool_name": name,
                    "tool_call_id": call_id,
                    "observation": obs,
                },
            ]
        )
        chats.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "result"},
            ]
        )
    events.extend(
        [
            {"kind": "TokenEvent", "prompt_token_ids": [1], "response_token_ids": [2]},
            {
                "kind": "TokenEvent",
                "prompt_token_ids": [1, 2, 3],
                "response_token_ids": [4],
            },
        ]
    )
    return {
        "instance_id": "case1",
        "status": "ok",
        "errors": [],
        "messages": events,
        "sft_messages": chats,
        "tools": [{"type": "function", "function": {"name": "grep"}}],
        "structured_locations": locations,
    }


def test_cleaning_dedup_quality_and_difficulty():
    valid = task()
    duplicate = {
        **valid,
        "instance_id": "case2",
        "problem_statement": "Repair   wrong addition",
    }
    blank = {**valid, "instance_id": "case3", "problem_statement": " "}
    rows, report = clean_tasks([valid, duplicate, blank])
    assert len(rows) == 1
    assert rows[0]["difficulty"] == "easy"
    assert len(report["rejected"]) == 2
    assert rows[0]["prompt"][0]["content"] == rows[0]["problem_statement"]


def test_failed_finish_or_unpaired_tool_is_invalid():
    row = trajectory()
    assert validate_trajectory(row) == []
    row["messages"][5]["observation"]["is_error"] = True
    assert validate_events(row["messages"])[0] is None
    assert "tool_error" in validate_trajectory(row)
    row = trajectory()
    row["messages"].pop(1)
    assert "unpaired_action" in validate_trajectory(row)


def test_existing_sample_tasks_with_nullable_target_lists():
    sample = Path(__file__).parents[1] / "data/sample/validation.parquet"
    original = pq.read_table(sample).to_pylist()
    kept, report = clean_tasks(original)
    assert len(kept) == len(original), report
    assert all(row["difficulty"] in {"easy", "medium", "hard"} for row in kept)


def test_malformed_tasks_do_not_abort_cleaning():
    invalid = task()
    invalid["file_changes"][0]["changes"] = "bad labels"
    kept, report = clean_tasks([None, invalid, {**task(), "use_patch": "false"}, task()])
    assert len(kept) == 1
    assert len(report["rejected"]) == 3


def test_malformed_event_is_filtered_without_breaking_metrics():
    row = trajectory()
    row["messages"] = [None, {"kind": "ObservationEvent", "observation": None}]
    assert validate_trajectory(row) == ["invalid_events"]
    assert tool_metrics(row["messages"])["tool_errors"] == 1


def test_invalid_action_schema_and_observation_order_are_rejected():
    row = trajectory()
    row["messages"][4]["action"]["locations"] = [{"file": "../outside.py"}]
    assert "invalid_action" in validate_trajectory(row)
    row = trajectory()
    row["messages"][0], row["messages"][1] = row["messages"][1], row["messages"][0]
    assert "observation_mismatch" in validate_trajectory(row)
    row = trajectory()
    row["messages"][0]["tool_call_id"] = []
    assert validate_trajectory(row) == ["invalid_tool_call_ids"]


def test_chat_arguments_must_match_executed_actions():
    row = trajectory()
    row["sft_messages"][1]["tool_calls"][0]["function"]["arguments"] = (
        '{"pattern":"other"}'
    )
    assert "chat_action_mismatch" in validate_trajectory(row)
    row["sft_messages"][1]["tool_calls"] = [None]
    assert validate_trajectory(row) == ["invalid_chat_tool_calls"]


def test_no_reward_for_unrelated_novel_content_and_cost_for_waste():
    row = trajectory()
    clean, _, _ = score_trajectory(task(), row["structured_locations"], row["messages"])
    wasted = row["messages"] + row["messages"][:4]
    lower, details, metrics = score_trajectory(
        task(), row["structured_locations"], wasted
    )
    assert lower < clean < 3
    assert metrics["repeated_searches"] == 1
    assert metrics["overlap_lines"] == 2
    assert details["file_reward"] == 1
    reward, _, _ = score_trajectory(task(), [{"file": "unrelated.py"}], wasted)
    assert reward == 0
    assert (
        tool_metrics(wasted)["output_chars"]
        > tool_metrics(row["messages"])["output_chars"]
    )


def test_export_roundtrip_rejects_bad_traces_and_duplicates(tmp_path):
    row = trajectory()
    broken = copy.deepcopy(row)
    broken["messages"][-1]["prompt_token_ids"] = [9]
    output = tmp_path / "export"
    report = export_data([task()], [row, row, broken], output, validation_fraction=0)
    assert report["sft_kept"] == 1
    assert len(report["rejected"]) == 2
    sft = pq.read_table(output / "sft/train.parquet").to_pylist()[0]
    assert sft["loss_mask"] == [0, 1, 0, 1]
    assert isinstance(sft["messages"], list)
    assert (
        pq.read_table(output / "rl/train.parquet").to_pylist()[0]["base_commit"]
        == "a" * 40
    )
    assert pq.read_table(output / "sft/validation.parquet").num_rows == 0


def test_evaluation_and_native_swe_bench_import(tmp_path):
    native = tmp_path / "swe.json"
    native.write_text(
        json.dumps({"resolved_ids": ["case1"], "unresolved_ids": ["unmatched"]})
    )
    report = evaluate([task()], [trajectory()], downstream=load_downstream(native))
    assert report["valid"] == 1
    assert report["means"]["file_reward"] == 1
    assert report["downstream"]["resolve_rate"] == 1
    assert report["downstream"]["unmatched_ids"] == ["unmatched"]
    with pytest.raises(ValueError, match="boolean"):
        downstream_summary([{"instance_id": "case1", "resolved": "false"}], {"case1"})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([task()], [trajectory(), trajectory()])
