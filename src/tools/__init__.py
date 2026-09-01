"""The fixed, read-only CodePin action space."""

from openhands.sdk.tool import Tool

from src.tools.glob import GlobTool
from src.tools.grep import GrepTool
from src.tools.localization_finish import LocalizationFinishTool
from src.tools.read_file import ReadFileTool

TOOL_NAMES = (
    GlobTool.name,
    GrepTool.name,
    ReadFileTool.name,
    LocalizationFinishTool.name,
)


def build_agent_tool_specs() -> list[Tool]:
    return [Tool(name=name) for name in TOOL_NAMES]


__all__ = [
    "TOOL_NAMES",
    "GlobTool",
    "GrepTool",
    "LocalizationFinishTool",
    "ReadFileTool",
    "build_agent_tool_specs",
]
