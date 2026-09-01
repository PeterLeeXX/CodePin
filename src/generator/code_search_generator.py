"""Multi-turn CodePin rollout generator for SkyRL."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ray
from openhands.sdk import LLM, Conversation, Event, LLMConvertibleEvent, get_logger
from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.sdk.event import ActionEvent
from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInterface,
)
from skyrl.train.config import GeneratorConfig
from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import get_rollout_metrics

from src.agent.agent import CustomAgent
from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
)
from src.tools import build_agent_tool_specs
from src.tools.localization_finish import LocalizationFinishAction
from src.utils.instance import clone_instance
from src.utils.trajectory_tokens import build_assistant_loss_mask

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "templates"
SYSTEM_PROMPT = PROMPTS_DIR / "system_prompt_atomic_search.j2"


def build_instruction(instance: dict[str, Any], working_dir: Path) -> str:
    return (
        f"Repository: {working_dir}\n\n"
        f"<issue_description>\n{instance['problem_statement']}\n"
        "</issue_description>\n\n"
        "Locate only the existing files, classes, and functions that must be "
        "modified. Finish with one localization_finish call."
    )


def get_structured_locations(events: list[Event]) -> list[dict[str, Any]] | None:
    finishes = [
        event
        for event in events
        if isinstance(event, ActionEvent)
        and event.source == "agent"
        and isinstance(event.action, LocalizationFinishAction)
    ]
    if len(finishes) != 1:
        return None
    return [location.model_dump() for location in finishes[0].action.locations]


def serialize_conversation(
    conversation: Conversation,
    agent: CustomAgent,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    list[dict[str, Any]] | None,
]:
    events = list(conversation.state.events)
    messages = [event.model_dump(mode="json") for event in events]
    llm_events = [event for event in events if isinstance(event, LLMConvertibleEvent)]
    sft_messages = [
        message.to_chat_dict()
        for message in LLMConvertibleEvent.events_to_messages(llm_events)
    ]
    tool_schemas = [
        json.loads(json.dumps(tool.to_openai_tool()))
        for tool in agent.tools_map.values()
    ]
    return (
        messages,
        sft_messages,
        tool_schemas,
        get_agent_final_response(events),
        get_structured_locations(events),
    )


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict[str, Any],
    model_name: str,
    base_url: str,
    generator_cfg: GeneratorConfig,
    sampling_params: dict[str, Any],
    max_tokens: int,
):
    workspace = Path("/tmp/testbed") / str(uuid.uuid4())[:8]
    conversation: Conversation | None = None
    agent: CustomAgent | None = None
    started = time.monotonic()
    empty: tuple[list, list, list, str, None] = ([], [], [], "", None)

    try:
        status, working_dir = clone_instance(
            instance["repo"],
            instance.get("base_commit"),
            instance["instance_id"],
            workspace,
            instance.get("patch") if instance.get("use_patch") else None,
        )
        if not status or working_dir is None:
            raise RuntimeError(f"Could not prepare {instance['instance_id']}")

        agent = CustomAgent(
            llm=LLM(
                usage_id="agent",
                model=model_name,
                base_url=base_url,
                api_key="sk-local",
                temperature=float(sampling_params.get("temperature", 1.0)),
                max_output_tokens=max_tokens,
                reasoning_effort="none",
                litellm_extra_body={
                    "return_token_ids": True,
                    "include_stop_str_in_output": False,
                    "top_k": sampling_params.get("top_k", 20),
                    "top_p": sampling_params.get("top_p", 1.0),
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
            max_iteration_per_run=generator_cfg.max_turns,
            visualizer=None,
            workspace=str(working_dir),
        )
        conversation.send_message(build_instruction(instance, working_dir))
        conversation.run()
        result = serialize_conversation(conversation, agent)
        return (*result, time.monotonic() - started)
    except Exception:
        logger.exception("Rollout failed for %s", instance.get("instance_id"))
        result = (
            serialize_conversation(conversation, agent)
            if conversation is not None and agent is not None
            else empty
        )
        return (*result, time.monotonic() - started)
    finally:
        if conversation is not None:
            try:
                conversation.close()
            except Exception:
                logger.exception("Could not close rollout conversation")
        if workspace.exists() and workspace.parent == Path("/tmp/testbed"):
            shutil.rmtree(workspace)


class CodeSearchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        policy_model_name: str,
    ):
        self.base_url = f"{inference_engine_client.get_endpoint_url().rstrip('/')}/v1"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = f"openai/{policy_model_name}"
        self.max_train_length = generator_cfg.max_train_length

    def valid_final_turn(self, token_messages: list[dict[str, Any]]) -> bool:
        if not token_messages:
            return False
        response = self.tokenizer.decode(
            token_messages[-1]["response_token_ids"], skip_special_tokens=False
        )
        if response.count("<tool_call>") != 1 or response.count("</tool_call>") != 1:
            return False
        if response.count("<|im_end|>") != 1:
            return False
        tail = response.split("</tool_call>", 1)[1].split("<|im_end|>", 1)[0]
        return not tail.strip()

    async def code_search_loop(
        self,
        _prompt: ConversationType,
        instance: dict[str, Any],
        max_tokens: int,
        sampling_params: dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ):
        error: str | None = None
        try:
            (
                messages,
                sft_messages,
                tool_schemas,
                final_message,
                locations,
                duration,
            ) = await init_and_run.remote(
                instance,
                self.model_name,
                self.base_url,
                self.generator_cfg,
                sampling_params,
                max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - record each rollout failure.
            error = f"{exc}\n{traceback.format_exc()}"
            messages, sft_messages, tool_schemas, final_message, locations = (
                [],
                [],
                [],
                "",
                None,
            )
            duration = 0.0

        token_messages = [m for m in messages if m.get("kind") == "TokenEvent"]
        exhausted = (
            locations is None
            and len(token_messages) >= self.generator_cfg.max_turns
        )
        if locations is not None and not self.valid_final_turn(token_messages):
            locations = None
            final_message = ""

        reward, reward_dict = multilevel_localization_f1_reward(
            instance=instance, structured_locations=locations
        )
        metrics = {
            "wall_clock_duration": duration,
            "num_turns": len(token_messages),
            "num_tool_calls": sum(
                message.get("kind") == "ActionEvent" for message in messages
            ),
        }

        if token_messages:
            try:
                prompt_ids, response_ids, loss_mask = build_assistant_loss_mask(
                    token_messages
                )
            except ValueError:
                prompt_ids = token_messages[0]["prompt_token_ids"]
                response_ids = token_messages[-1]["response_token_ids"]
                loss_mask = [0] * len(response_ids)
            limit = max(0, self.max_train_length - len(prompt_ids))
            response_ids = response_ids[:limit]
            loss_mask = loss_mask[:limit]
            if exhausted:
                loss_mask = [0] * len(loss_mask)
            rollout = (
                response_ids,
                reward,
                "complete",
                loss_mask,
                prompt_ids,
                None,
                metrics,
            )
        else:
            eos = self.tokenizer.eos_token_id or 0
            rollout = ([eos], reward, "error", [0], [eos], None, metrics)

        output_dir = (
            Path(self.generator_cfg.traj_dir)
            / f"step_{batch_metadata.global_step}"
            / str(batch_metadata.training_phase)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{instance['instance_id']}_{trajectory_id.repetition_id}"
        if error:
            (output_dir / f"{stem}.error").write_text(error, encoding="utf-8")
        else:
            payload = {
                "instance_id": instance["instance_id"],
                "target": instance["target"],
                "total_reward": reward,
                "reward_dict": reward_dict,
                "structured_locations": locations,
                "final_message": final_message,
                "messages": messages,
                "sft_messages": sft_messages,
                "tools": tool_schemas,
                "metrics": metrics,
            }
            (output_dir / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return [rollout], reward_dict, metrics

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        instances = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids")
        batch_metadata = input_batch.get("batch_metadata")
        if trajectory_ids is None or batch_metadata is None:
            raise ValueError("trajectory_ids and batch_metadata are required")

        sampling_params = input_batch.get("sampling_params") or asdict(
            self.generator_cfg.sampling_params
        )
        tasks = [
            self.code_search_loop(
                prompt,
                instance,
                self.generator_cfg.sampling_params.max_generate_length,
                sampling_params,
                trajectory_id,
                batch_metadata,
            )
            for prompt, instance, trajectory_id in zip(
                prompts, instances, trajectory_ids, strict=True
            )
        ]
        collected = await asyncio.gather(*tasks)
        outputs = [item[0][0] for item in collected]
        reward_details = [item[1] for item in collected]
        metrics = [item[2] for item in collected]

        responses = [item[0] for item in outputs]
        rewards = [item[1] for item in outputs]
        stop_reasons = [item[2] for item in outputs]
        loss_masks = [item[3] for item in outputs]
        prompt_ids = [item[4] for item in outputs]
        tracked: dict[str, float] = {}
        for prefix, rows in (("reward", reward_details), ("metrics", metrics)):
            keys = {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (int, float))
            }
            for key in keys:
                values = [float(row[key]) for row in rows if key in row]
                tracked[f"{prefix}/{key}"] = sum(values) / len(values)

        return {
            "trajectory_ids": trajectory_ids,
            "prompt_token_ids": prompt_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": get_rollout_metrics(
                responses, rewards, loss_masks=loss_masks
            ),
            "rollout_logprobs": None,
            "trajectory_generation_times": [
                row["wall_clock_duration"] for row in metrics
            ],
            "rollout_expert_indices": None,
            "is_last_step": None,
            "env_metrics": metrics,
            "pixel_values": None,
            "image_grid_thw": None,
            **tracked,
        }
