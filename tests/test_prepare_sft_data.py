import json

from scripts.prepare_sft_data import normalized_row


def _function_tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _atomic_tools():
    return [
        _function_tool(
            "glob",
            {"pattern": {"type": "string"}, "path": {"type": "string"}},
            ["pattern"],
        ),
        _function_tool(
            "grep",
            {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include": {"type": "string"},
            },
            ["pattern"],
        ),
        _function_tool(
            "read_file",
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            ["path"],
        ),
        _function_tool(
            "localization_finish",
            {
                "locations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "class_name": {"type": "string"},
                            "function_name": {"type": "string"},
                        },
                        "required": ["file"],
                    },
                }
            },
            ["locations"],
        ),
    ]


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
                "tools": _atomic_tools(),
            }
        ),
        encoding="utf-8",
    )

    row = normalized_row(path, set(), 1.0)

    assert row is not None
    assert row["instance_id"] == "task-1"
    assert row["messages"][-1]["role"] == "assistant"
    assert [tool["function"]["name"] for tool in row["tools"]] == [
        "glob",
        "grep",
        "read_file",
        "localization_finish",
    ]


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


def test_normalized_row_rejects_legacy_terminal_schema(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": "legacy-task",
                "total_reward": 1.0,
                "sft_messages": [{"role": "assistant", "content": "done"}],
                "tools": [{"type": "function", "function": {"name": "terminal"}}],
            }
        ),
        encoding="utf-8",
    )

    assert normalized_row(path, set(), 1.0) is None


def test_normalized_row_rejects_same_names_with_incompatible_fields(tmp_path):
    tools = _atomic_tools()
    tools[1]["function"]["parameters"]["properties"]["output_mode"] = {"type": "string"}
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": "wrong-schema",
                "total_reward": 1.0,
                "sft_messages": [{"role": "assistant", "content": "done"}],
                "tools": tools,
            }
        ),
        encoding="utf-8",
    )

    assert normalized_row(path, set(), 1.0) is None
