import json

from scripts.generate_teacher_trajectories import (
    artifact_name,
    attempt_temperature,
    canonical_tools,
    execute_glob,
    execute_grep,
    execute_read_file,
    ground_truth_levels,
    normalize_sft_messages,
    retry_strategy,
    score_locations,
    stratified_split,
    validate_saved_record,
)


def sample_row():
    return {
        "instance_id": "owner__repo.func_bug__123",
        "file_changes": [
            {
                "file": "src/service.py",
                "changes": {
                    "added_entities": None,
                    "added_modules": None,
                    "edited_modules": ["src/service.py:Service"],
                    "edited_entities": ["src/service.py:Service.run"],
                },
            }
        ],
    }


def test_ground_truth_and_exact_multilevel_score():
    row = sample_row()
    assert ground_truth_levels(row) == {
        "files": {"src/service.py"},
        "modules": {"src/service.py:Service"},
        "entities": {"src/service.py:Service.run"},
    }

    score = score_locations(
        [
            {
                "file": "src/service.py",
                "class_name": "Service",
                "function_name": "run",
            }
        ],
        row,
    )
    assert score["perfect"] is True
    assert score["file_reward"] == 1.0
    assert score["module_reward"] == 1.0
    assert score["entity_reward"] == 1.0


def test_file_only_prediction_does_not_pass_symbol_target():
    score = score_locations([{"file": "src/service.py"}], sample_row())
    assert score["exact"]["files"] is True
    assert score["perfect"] is False


def test_local_tools_are_bounded_and_repository_relative(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "service.py").write_text(
        "class Service:\n    def run(self):\n        return 1\n", encoding="utf-8"
    )
    (source / "other.txt").write_text("needle\n", encoding="utf-8")

    assert execute_glob(tmp_path, {"pattern": "**/*.py"}) == "src/service.py"
    grep = execute_grep(tmp_path, {"pattern": "def run", "include": "*.py"})
    assert "src/service.py:2:" in grep
    read = execute_read_file(
        tmp_path, {"path": "src/service.py", "start_line": 1, "end_line": 2}
    )
    assert "lines 1-2 of 3" in read
    assert "def run" in read


def test_saved_record_requires_one_observed_finish_call():
    arguments = json.dumps(
        {
            "locations": [
                {
                    "file": "src/service.py",
                    "class_name": "Service",
                    "function_name": "run",
                }
            ]
        }
    )
    record = {
        "accepted": True,
        "reward_dict": {"perfect": True},
        "tools": canonical_tools(),
        "sft_messages": [
            {"role": "system", "content": "search"},
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "localization_finish",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "localization_finish",
                "content": "ok",
            },
        ],
    }
    assert validate_saved_record(record) == []

    record["sft_messages"][-1]["tool_call_id"] = "wrong"
    assert "tool_call_observation_mismatch" in validate_saved_record(record)


def test_normalize_sft_messages_uses_qwen_argument_objects():
    messages = [
        {
            "role": "assistant",
            "content": "I found the target.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "localization_finish",
                        "arguments": json.dumps({"locations": [{"file": "src/service.py"}]}),
                    },
                }
            ],
        }
    ]

    normalized = normalize_sft_messages(messages)

    assert normalized[0]["content"] == "I found the target."
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
        "locations": [{"file": "src/service.py"}]
    }
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)


def test_artifact_names_are_stable_and_do_not_create_subdirectories():
    first = artifact_name("repo/task with spaces")
    second = artifact_name("repo/task with spaces")
    assert first == second
    assert "/" not in first
    assert first.endswith(".json")


def test_retry_attempts_vary_strategy_and_temperature():
    assert attempt_temperature(0.2, 1) == 0.2
    assert attempt_temperature(0.2, 2) == 0.0
    assert attempt_temperature(0.2, 3) == 0.15
    assert retry_strategy(1) == ""
    assert retry_strategy(2)
    assert retry_strategy(2) != retry_strategy(3)


def test_stratified_split_preserves_joint_groups():
    records = []
    for bucket in ("representative", "boundary"):
        for difficulty in ("easy", "hard"):
            for index in range(20):
                records.append(
                    {
                        "instance_id": f"{bucket}-{difficulty}-{index}",
                        "selection_metadata": {
                            "selection_bucket": bucket,
                            "difficulty": difficulty,
                        },
                    }
                )

    train, validation = stratified_split(records, seed=42, validation_ratio=0.1)

    assert len(train) == 72
    assert len(validation) == 8
    assert {
        (
            row["selection_metadata"]["selection_bucket"],
            row["selection_metadata"]["difficulty"],
        )
        for row in validation
    } == {
        ("representative", "easy"),
        ("representative", "hard"),
        ("boundary", "easy"),
        ("boundary", "hard"),
    }
