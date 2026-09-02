"""One read-only OpenHands execution path shared by serving and RL."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openhands.sdk import LLM, Conversation, LLMConvertibleEvent
from openhands.sdk.conversation.response_utils import get_agent_final_response

from src.agent.agent import CustomAgent
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
    result = {
        "messages": [],
        "sft_messages": [],
        "tools": [],
        "final_message": "",
        "structured_locations": None,
        "errors": [],
    }
    try:
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
        conversation.run()
        result = serialize_conversation(conversation, agent)
    except Exception as exc:  # noqa: BLE001 - return explicit failures to every caller.
        if conversation is not None:
            result = serialize_conversation(conversation, agent)
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["structured_locations"] = None
    finally:
        if conversation is not None:
            conversation.close()
    result["metrics"] = tool_metrics(result["messages"])
    result["metrics"]["wall_clock_duration"] = time.monotonic() - started
    result["status"] = "ok" if not result["errors"] else "error"
    return result
