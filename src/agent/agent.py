"""OpenHands agent restricted to CodePin's explicit tool set."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from openhands.sdk import Agent
from openhands.sdk.logger import get_logger
from openhands.sdk.mcp import create_mcp_tools
from openhands.sdk.tool import ToolDefinition, resolve_tool

from src.profiling import nvtx_range
from src.tools import initialize_tool_schemas

if TYPE_CHECKING:
    from openhands.sdk.conversation import ConversationState


logger = get_logger(__name__)


class CustomAgent(Agent):
    """Resolve only configured tools instead of adding SDK built-ins."""

    def step(self, conversation, on_event, on_token=None) -> None:
        state = conversation.state
        with nvtx_range(f"codepin.step|{state.id}|events={len(state.events)}"):
            super().step(conversation, on_event, on_token)

    @nvtx_range("codepin.agent_initialize")
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


# Rebuild after this last SDK subclass is defined, while imports are serialized.
# Lazy rebuilding during a concurrent ConversationState constructor is unsafe.
initialize_tool_schemas()
CustomAgent.model_json_schema()
