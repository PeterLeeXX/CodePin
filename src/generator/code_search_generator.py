"""Multi-turn CodePin rollout generator for SkyRL."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ray
from openhands.sdk import get_logger
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
from skyrl.train.generators.utils import apply_overlong_filtering, get_rollout_metrics

from src.rollout import run_localization
from src.trajectory import score_trajectory
from src.utils.instance import clone_instance
from src.utils.trajectory_tokens import build_assistant_loss_mask

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)
# Keep SkyRL's step metrics at INFO while suppressing the SDK's per-conversation
# state snapshots, which otherwise serialize large messages from every rollout
# into the shared Ray log stream.
logging.getLogger("openhands").setLevel(logging.WARNING)


# Each rollout owns a Python/OpenHands process and performs git/file I/O.  The
# previous fractional reservation let large batches spawn hundreds of workers
# at once, overwhelming Raylet before vLLM could serve them.  Reserve one CPU
# so Ray provides bounded, host-aware rollout concurrency.
@ray.remote(num_cpus=1)
def init_and_run(
    instance: dict[str, Any],
    model_name: str,
    base_url: str,
    generator_cfg: GeneratorConfig,
    sampling_params: dict[str, Any],
    max_tokens: int,
):
    workspace = Path("/tmp/testbed") / str(uuid.uuid4())[:8]
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
        return run_localization(
            instance,
            working_dir,
            model=model_name,
            base_url=base_url,
            max_turns=generator_cfg.max_turns,
            max_tokens=max_tokens,
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", 20)),
        )
    finally:
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
        if generator_cfg.result_cache:
            raise ValueError("training rollout must disable result_cache")
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
            result = await init_and_run.remote(
                instance,
                self.model_name,
                self.base_url,
                self.generator_cfg,
                sampling_params,
                max_tokens,
            )
            messages = result["messages"]
            sft_messages = result["sft_messages"]
            tool_schemas = result["tools"]
            final_message = result["final_message"]
            locations = result["structured_locations"]
            duration = result["metrics"]["wall_clock_duration"]
            error = "; ".join(result["errors"]) or None
        except Exception as exc:  # noqa: BLE001 - persist worker errors as invalid rollouts.
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
            locations is None and len(token_messages) >= self.generator_cfg.max_turns
        )
        if locations is not None and not self.valid_final_turn(token_messages):
            locations = None
            final_message = ""

        reward, reward_dict, metrics = score_trajectory(
            instance,
            locations,
            messages,
            efficiency_weight=self.generator_cfg.efficiency_weight,
            valid=locations is not None and error is None,
        )
        metrics["wall_clock_duration"] = duration

        if token_messages:
            try:
                prompt_ids, response_ids, loss_mask = build_assistant_loss_mask(
                    token_messages
                )
            except ValueError as exc:
                error = f"invalid_token_trace: {exc}"
                prompt_ids = token_messages[0]["prompt_token_ids"]
                response_ids = token_messages[-1]["response_token_ids"]
                loss_mask = [0] * len(response_ids)
            limit = max(0, self.max_train_length - len(prompt_ids))
            truncated = len(response_ids) > limit
            response_ids = response_ids[:limit]
            loss_mask = loss_mask[:limit]
            if exhausted or error or locations is None or truncated:
                loss_mask = [0] * len(loss_mask)
            stop_reason = (
                "stop"
                if locations is not None and not error and not truncated
                else "length"
                if exhausted or truncated
                else "error"
            )
            if stop_reason != "stop":
                reward = 0.0
                reward_dict.update(total_reward=0.0, trajectory_valid=0.0)
                if not response_ids:
                    response_ids = [self.tokenizer.eos_token_id or 0]
                    loss_mask = [0]
                    prompt_ids = prompt_ids[: max(1, self.max_train_length - 1)]
            metrics["loss_tokens"] = sum(loss_mask)
            metrics["invalid_trajectory"] = float(stop_reason != "stop")
            rollout = (
                response_ids,
                reward,
                stop_reason,
                loss_mask,
                prompt_ids,
                None,
                metrics,
            )
        else:
            eos = self.tokenizer.eos_token_id or 0
            rollout = ([eos], reward, "error", [0], [eos], None, metrics)

        metrics["loss_tokens"] = sum(rollout[3])
        metrics["invalid_trajectory"] = float(rollout[2] != "stop")

        output_dir = (
            Path(self.generator_cfg.traj_dir)
            / f"step_{batch_metadata.global_step}"
            / str(batch_metadata.training_phase)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{instance['instance_id']}_{trajectory_id.repetition_id}"
        payload = {
            "instance_id": instance["instance_id"],
            "target": instance.get("target", instance.get("file_changes")),
            "total_reward": reward,
            "reward_dict": reward_dict,
            "structured_locations": locations,
            "final_message": final_message,
            "messages": messages,
            "sft_messages": sft_messages,
            "tools": tool_schemas,
            "metrics": metrics,
            "errors": [error] if error else [],
            "status": rollout[2],
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
        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)
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
