from types import SimpleNamespace

from src.tools import (
    GlobTool,
    GrepTool,
    LocalizationFinishTool,
    ReadFileTool,
    build_agent_tool_specs,
)


def test_atomic_toolset_is_fixed_and_ordered():
    assert [spec.name for spec in build_agent_tool_specs()] == [
        "glob",
        "grep",
        "read_file",
        "localization_finish",
    ]


def test_all_tool_schemas_are_read_only(tmp_path):
    state = SimpleNamespace(workspace=SimpleNamespace(working_dir=str(tmp_path)))
    definitions = [
        tool.create(state)[0]
        for tool in (GlobTool, GrepTool, ReadFileTool, LocalizationFinishTool)
    ]
    assert all(definition.annotations.readOnlyHint for definition in definitions)
