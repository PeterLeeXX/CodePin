import json

from scripts.prepare_sft_data import normalized_row


def test_normalized_row_keeps_successful_openai_chat(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "total_reward": 1.0,
                "sft_messages": [
                    {"role": "user", "content": "find it"},
                    {"role": "assistant", "content": "done"},
                ],
                "tools": [{"type": "function", "function": {"name": "terminal"}}],
            }
        ),
        encoding="utf-8",
    )

    row = normalized_row(path, set(), 1.0)

    assert row is not None
    assert row["instance_id"] == "task-1"
    assert row["messages"][-1]["role"] == "assistant"


def test_normalized_row_filters_reward_and_reserved_instances(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": "reserved",
                "total_reward": 0.5,
                "sft_messages": [{"role": "assistant", "content": "x"}],
            }
        ),
        encoding="utf-8",
    )

    assert normalized_row(path, set(), 1.0) is None
    assert normalized_row(path, {"reserved"}, 0.0) is None
