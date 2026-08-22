"""CodePin-specific OpenHands tools."""

from openhands.sdk.tool import Tool

from src.tools.glob import GlobTool
from src.tools.grep import GrepTool
from src.tools.localization_finish import LocalizationFinishTool
from src.tools.read_file import ReadFileTool

SEARCH_TOOL_NAMES = (GlobTool.name, GrepTool.name, ReadFileTool.name)


def build_agent_tool_specs(configured_tools: list[str]) -> list[Tool]:
    """Build the canonical tool set and reject incompatible configurations."""
    names = list(configured_tools)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate generator tools are not allowed: {names}")

    expected = set(SEARCH_TOOL_NAMES)
    actual = set(names)
    if actual != expected:
        missing = sorted(expected - actual)
        unsupported = sorted(actual - expected)
        raise ValueError(
            "CodePin requires exactly the atomic search tools "
            f"{list(SEARCH_TOOL_NAMES)}; missing={missing}, unsupported={unsupported}"
        )

    return [Tool(name=name) for name in SEARCH_TOOL_NAMES] + [
        Tool(name=LocalizationFinishTool.name)
    ]


__all__ = [
    "GlobTool",
    "GrepTool",
    "LocalizationFinishTool",
    "ReadFileTool",
    "build_agent_tool_specs",
]
