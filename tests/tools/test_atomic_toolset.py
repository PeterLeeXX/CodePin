from types import SimpleNamespace

import pytest

from scripts.prepare_sft_data import has_atomic_tool_schema
from src.tools import (
    GlobTool,
    GrepTool,
    LocalizationFinishTool,
    ReadFileTool,
    build_agent_tool_specs,
)


def test_atomic_toolset_contains_exactly_four_tools():
    specs = build_agent_tool_specs(["glob", "grep", "read_file"])

    assert [spec.name for spec in specs] == [
        "glob",
        "grep",
        "read_file",
        "localization_finish",
    ]


def test_atomic_toolset_order_is_stable():
    specs = build_agent_tool_specs(["read_file", "glob", "grep"])

    assert [spec.name for spec in specs] == [
        "glob",
        "grep",
        "read_file",
        "localization_finish",
    ]


def test_generated_tool_schemas_pass_sft_contract(tmp_path):
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=str(tmp_path)))
    definitions = [
        tool_type.create(state)[0]
        for tool_type in (GlobTool, GrepTool, ReadFileTool, LocalizationFinishTool)
    ]
    schemas = [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "parameters": definition.to_openai_tool().function.parameters,
            },
        }
        for definition in definitions
    ]

    assert has_atomic_tool_schema(schemas)


@pytest.mark.parametrize(
    "configured",
    [
        ["glob", "grep"],
        ["glob", "grep", "terminal"],
        ["glob", "grep", "read_file", "read_file"],
    ],
)
def test_atomic_toolset_rejects_unstable_configurations(configured):
    with pytest.raises(ValueError):
        build_agent_tool_specs(configured)
