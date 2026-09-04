"""The fixed, read-only CodePin action space."""

from openhands.sdk.tool import Tool
from openhands.sdk.tool.tool import create_action_type_with_risk

from src.tools.glob import GlobAction, GlobTool
from src.tools.grep import GrepAction, GrepTool
from src.tools.localization_finish import (
    LocalizationFinishAction,
    LocalizationFinishTool,
)
from src.tools.read_file import ReadFileAction, ReadFileTool

TOOL_NAMES = (
    GlobTool.name,
    GrepTool.name,
    ReadFileTool.name,
    LocalizationFinishTool.name,
)


def initialize_tool_schemas() -> None:
    """Register native SDK variants before concurrent conversations start."""
    for action in (GlobAction, GrepAction, ReadFileAction, LocalizationFinishAction):
        create_action_type_with_risk(action)


def build_agent_tool_specs() -> list[Tool]:
    return [Tool(name=name) for name in TOOL_NAMES]


__all__ = [
    "TOOL_NAMES",
    "GlobTool",
    "GrepTool",
    "LocalizationFinishTool",
    "ReadFileTool",
    "build_agent_tool_specs",
    "initialize_tool_schemas",
]
