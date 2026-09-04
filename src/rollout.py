"""One read-only OpenHands execution path shared by serving and RL."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from openhands.sdk import LLM, Conversation, LLMConvertibleEvent
from openhands.sdk.conversation.response_utils import get_agent_final_response

from src.agent.agent import CustomAgent
from src.profiling import issue_trace_id, measure_stage
from src.tools import build_agent_tool_specs
from src.trajectory import tool_metrics, validate_events

SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts/templates/system_prompt_atomic_search.j2"
)
logging.getLogger("openhands").setLevel(logging.WARNING)


def build_instruction(instance: dict, working_dir: Path) -> str:
    return (
        f"Repository: {working_dir}\n\n"
        f"<issue_description>\n{instance['problem_statement']}\n"
        "</issue_description>\n\n"
        "Locate only the existing files, classes, and functions that must be "
        "modified. Finish with one localization_finish call."
    )


def serialize_conversation(conversation: Conversation, agent: CustomAgent) -> dict:
    events = list(conversation.state.events)
    messages = [event.model_dump(mode="json") for event in events]
    llm_events = [event for event in events if isinstance(event, LLMConvertibleEvent)]
    locations, errors = validate_events(messages)
    return {
        "messages": messages,
        "sft_messages": [
            m.to_chat_dict() for m in LLMConvertibleEvent.events_to_messages(llm_events)
        ],
        "tools": [
            json.loads(json.dumps(tool.to_openai_tool()))
            for tool in agent.tools_map.values()
        ],
        "final_message": get_agent_final_response(events),
        "structured_locations": locations,
        "errors": errors,
    }


def run_localization(
    instance: dict,
    working_dir: Path,
    *,
    model: str,
    base_url: str,
    max_turns: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 20,
) -> dict:
    """Run an uncached trajectory. Exceptions remain explicit and untrainable."""
    started = time.monotonic()
    conversation = None
    stages = {}
    result = {
        "messages": [],
        "sft_messages": [],
        "tools": [],
        "final_message": "",
        "structured_locations": None,
        "errors": [],
    }
    try:
        stage_started = time.monotonic()
        agent = CustomAgent(
            llm=LLM(
                usage_id="agent",
                model=model,
                base_url=base_url,
                api_key="sk-local",
                temperature=temperature,
                max_output_tokens=max_tokens,
                reasoning_effort="none",
                litellm_extra_body={
                    "return_token_ids": True,
                    "include_stop_str_in_output": False,
                    "top_k": top_k,
                    "top_p": top_p,
                    "chat_template_kwargs": {
                        "add_generation_prompt": True,
                        "enable_thinking": False,
                    },
                },
            ),
            tools=build_agent_tool_specs(),
            system_prompt_filename=str(SYSTEM_PROMPT),
        )
        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=max_turns,
            visualizer=None,
            workspace=str(working_dir),
        )
        conversation.send_message(build_instruction(instance, working_dir))
        stages["agent_setup_seconds"] = time.monotonic() - stage_started
        with measure_stage(stages, "conversation_run_seconds"):
            conversation.run()
        with measure_stage(stages, "conversation_serialize_seconds"):
            result = serialize_conversation(conversation, agent)
    except Exception as exc:  # noqa: BLE001 - return explicit failures to every caller.
        if conversation is not None:
            with measure_stage(stages, "conversation_serialize_seconds"):
                result = serialize_conversation(conversation, agent)
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["structured_locations"] = None
    finally:
        if conversation is not None:
            with measure_stage(stages, "conversation_close_seconds"):
                conversation.close()
    result["metrics"] = tool_metrics(result["messages"])
    prompt_lengths = [
        len(event.get("prompt_token_ids") or [])
        for event in result["messages"]
        if event.get("kind") == "TokenEvent"
    ]
    result["metrics"]["prompt_tokens"] = sum(prompt_lengths)
    result["metrics"]["max_prompt_tokens"] = max(prompt_lengths, default=0)
    result["metrics"].update(stages)
    result["metrics"]["wall_clock_duration"] = time.monotonic() - started
    result["status"] = "ok" if not result["errors"] else "error"
    result["execution_id"] = str(conversation.state.id) if conversation else None
    if (trace_directory := os.environ.get("CODEPIN_PERF_TRACE_DIR")) and conversation:
        trace_path = Path(trace_directory)
        trace_path.mkdir(parents=True, exist_ok=True)
        (trace_path / f"{conversation.state.id}.json").write_text(
            json.dumps(
                {
                    "conversation_id": str(conversation.state.id),
                    "instance_id": instance.get("instance_id")
                    or str(conversation.state.id),
                    "issue_id": issue_trace_id(instance["problem_statement"]),
                    "instance": instance,
                    "repository": str(working_dir),
                    **result,
                }
            ),
            encoding="utf-8",
        )
    return result
