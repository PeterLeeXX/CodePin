import subprocess
import sys
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


def test_cold_concurrent_agent_schema_initialization(tmp_path):
    # A fresh interpreter is essential: prior tests warm the SDK schema registry.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from openhands.sdk import LLM, Conversation
from src.agent.agent import CustomAgent
from src.rollout import SYSTEM_PROMPT
from src.tools import TOOL_NAMES, build_agent_tool_specs

barrier = Barrier(16)
def initialize(_):
    barrier.wait(timeout=30)
    agent = CustomAgent(
        llm=LLM(model='openai/codepin', api_key='local', usage_id='test'),
        tools=build_agent_tool_specs(),
        system_prompt_filename=str(SYSTEM_PROMPT),
    )
    conversation = Conversation(
        agent=agent, workspace=sys.argv[1], visualizer=None,
    )
    try:
        assert conversation.state.security_analyzer is None
        assert conversation.state.agent is agent
        agent._initialize(conversation.state)
        assert tuple(agent.tools_map) == TOOL_NAMES
        return [tool.to_openai_tool() for tool in agent.tools_map.values()]
    finally:
        conversation.close()

with ThreadPoolExecutor(max_workers=16) as pool:
    schemas = list(pool.map(initialize, range(64)))
assert all(schema == schemas[0] for schema in schemas)
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
