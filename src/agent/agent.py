"""OpenHands agent restricted to CodePin's explicit tool set."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from openhands.sdk import Agent
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.tool import ToolDefinition, resolve_tool

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState


logger = get_logger(__name__)


class CustomAgent(Agent):
    """Resolve only configured tools instead of adding SDK built-ins."""

    def _initialize(self, state: ConversationState) -> None:
        if self._tools:
            return

        tools: list[ToolDefinition] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(resolve_tool, tool_spec, state)
                for tool_spec in self.tools
            ]
            if self.mcp_config:
                futures.append(executor.submit(create_mcp_tools, self.mcp_config, 30))
            for future in futures:
                tools.extend(future.result())

        if self.filter_tools_regex:
            pattern = re.compile(self.filter_tools_regex)
            tools = [tool for tool in tools if pattern.match(tool.name)]

        if any(not isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("All resolved tools must be ToolDefinition instances")

        names = [tool.name for tool in tools]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate tool names: {sorted(duplicates)}")

        logger.info("Loaded CodePin tools: %s", names)
        self._tools = {tool.name: tool for tool in tools}
