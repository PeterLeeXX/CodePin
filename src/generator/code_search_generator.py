import asyncio
import json
import logging
import os
import re
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import gcsfs
import ray
from openhands.sdk import (
    LLM,
    Conversation,
    Event,
    LLMConvertibleEvent,
    get_logger,
)
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
    TrainingPhase,
    TrajectoryID,
)
from skyrl.train.generators.utils import get_rollout_metrics

from src.agent.agent import CustomAgent
from src.metrics.efficiency_metrics import compute_all_efficiency_metrics
from src.metrics.trajectory_metrics import compute_trajectory_metrics
from src.prompts.prompt_builder import get_instruction
from src.rewards import get_reward_function
from src.tools import build_agent_tool_specs
from src.tools.localization_finish import LocalizationFinishAction
from src.utils.instance import clone_instance
from src.utils.trajectory_tokens import build_assistant_loss_mask

logger = get_logger(__name__)
logger.setLevel(logging.ERROR)

file_path = os.path.dirname(__file__)

def get_structured_locations(events: List[Event]) -> Optional[List[Dict[str, Any]]]:
    """Extract structured locations from LocalizationFinishAction in events.
    Args:
        events: List of conversation events to search through.
    Returns:
        List of location dicts with 'file', 'class', 'function' keys, or None if not found.
    """
    # Find the last LocalizationFinishAction
    cnt = [
        1
        for event in events
        if isinstance(event, ActionEvent)
        and event.source == "agent"
        and isinstance(event.action, LocalizationFinishAction)
    ]
    cnt = sum(cnt)
    if cnt != 1:  # the localization finish tool must be called exactly once.
        return None
    for event in reversed(events):
        if (
            isinstance(event, ActionEvent)
            and event.source == "agent"
            and isinstance(event.action, LocalizationFinishAction)
        ):
            # Extract structured locations from the action
            locations = []
            for loc in event.action.locations:
                locations.append(
                    {
                        "file": loc.file,
                        "class_name": loc.class_name,
                        "function_name": loc.function_name,
                    }
                )
            return locations
    return None


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: str,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    max_tokens: int,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):

    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance.get("base_commit", None)
    if "use_patch" in instance and instance["use_patch"]:
        patch = instance["patch"]
    else:
        patch = None

    # Avoid collisions in /tmp testbed directories
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    status, working_dir = clone_instance(
        repo_name, commit_id, instance_id, workspace, patch
    )

    temperature = float(sampling_params.get("temperature", 1.0))

    final_message = ""
    structured_locations = None
    messages = []
    sft_messages = []
    tool_schemas = []

    tools = build_agent_tool_specs(generator_cfg.tools)

    # Get prompt paths from config (path-independent)
    prompts_base_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    system_prompt_path = os.path.join(
        prompts_base_dir, generator_cfg.prompts.system_prompt
    )
    user_prompt_path = os.path.join(prompts_base_dir, generator_cfg.prompts.user_prompt)

    assert os.path.exists(system_prompt_path), (
        f"System prompt file {system_prompt_path} does not exist"
    )
    assert os.path.exists(user_prompt_path), (
        f"User prompt file {user_prompt_path} does not exist"
    )

    agent = CustomAgent(
        llm=LLM(
            usage_id="agent",
            model=litellm_model_name,
            base_url=litellm_base_url,
            api_key="sk-xxx",
            temperature=temperature,
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
        tools=tools,
        system_prompt_filename=system_prompt_path,
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=generator_cfg.max_turns,
        visualizer=None,
        workspace=str(working_dir),
    )
    input_message = get_instruction(instance, user_prompt_path, str(working_dir))

    # Capture start time
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    try:
        conversation.send_message(input_message)
        logger.info("Conversation Starting")
        conversation.run()
        events = list(conversation.state.events)
        messages = [event.model_dump(mode="json") for event in events]
        llm_events = [
            event for event in events if isinstance(event, LLMConvertibleEvent)
        ]
        sft_messages = [
            message.to_chat_dict()
            for message in LLMConvertibleEvent.events_to_messages(llm_events)
        ]
        tool_schemas = [
            json.loads(json.dumps(tool.to_openai_tool()))
            for tool in agent.tools_map.values()
        ]
        final_message = get_agent_final_response(conversation.state.events)
        structured_locations = get_structured_locations(conversation.state.events)
    except Exception as e:
        logger.error(f"Error during conversation: {str(e)}", exc_info=True)
        try:
            events = list(conversation.state.events)
            messages = [event.model_dump(mode="json") for event in events]
            llm_events = [
                event for event in events if isinstance(event, LLMConvertibleEvent)
            ]
            sft_messages = [
                message.to_chat_dict()
                for message in LLMConvertibleEvent.events_to_messages(llm_events)
            ]
            tool_schemas = [
                json.loads(json.dumps(tool.to_openai_tool()))
                for tool in agent.tools_map.values()
            ]
            final_message = get_agent_final_response(conversation.state.events)
            structured_locations = get_structured_locations(conversation.state.events)
        except Exception as e:
            logger.error(
                f"Error during final message extraction in err'ed rollout: {str(e)}",
                exc_info=True,
            )
            messages = []
            final_message = ""
    finally:
        # Capture end time
        try:
            if workspace.exists() and workspace.parent == Path("/tmp/testbed"):
                shutil.rmtree(workspace)
                logger.info(f"Removed workspace {str(workspace)}")
            conversation.close()
        except Exception as _:
            pass
        logger.info("Conversation Finished")
        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        wall_clock_duration = end_time - start_time

        additional_attr = {
            "wall_clock_duration": wall_clock_duration,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        }

    # NOTE: Hard-coded final message to ensure all rollouts that don't call the custom finish tool have reward == 0
    return (
        messages,
        sft_messages,
        tool_schemas,
        final_message,
        structured_locations,
        additional_attr,
    )


class CodeSearchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        policy_model_name: str,
    ):
        self.base_url = f"{inference_engine_client.get_endpoint_url().rstrip('/')}/v1"
        logger.info(
            f"Using CodeSearchGenerator with model {policy_model_name} at {self.base_url}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = policy_model_name
        self.litellm_model_name = "openai/" + self.model_name
        self.step_wise = generator_cfg.step_wise_trajectories
        self.max_train_length = generator_cfg.max_train_length

    def sanity_check_last_step(self, token_messages):
        # Checks if the tool call formatting is correct in the last step from the detokenized response str of the last turn
        if len(token_messages) == 0:
            return False
        response_token_ids = token_messages[-1]["response_token_ids"]
        last_response_str: str = self.tokenizer.decode(
            response_token_ids, skip_special_tokens=False
        )
        # First sanity check -- verify if there is exactly one <tool_call> and one </tool_call> in response (if there are multiple tool calls give 0 reward regardless of correctness)
        cnt_tool_call = last_response_str.count("<tool_call>")
        cnt_tool_end = last_response_str.count("</tool_call>")
        if cnt_tool_call != 1 or cnt_tool_end != 1:
            return False
        # Second sanity check -- verify if the <|im_end|> is present exactly once
        elif last_response_str.count("<|im_end|>") != 1:
            return False
        # Third sanity check -- verify if there is no non-whitespace text after </tool_call> and before <|im_end|>
        else:
            portion = last_response_str.split("</tool_call>")[1].split("<|im_end|>")[0]
            if portion.strip() != "":
                return False
        return True

    async def code_search_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[
        List[int],
        float,
        str,
        List[int],
        List[int],
        Optional[List[int]],
        Optional[Dict[str, Any]],
    ]:
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras
        error = None
        try:
            (
                messages,
                sft_messages,
                tool_schemas,
                final_message,
                structured_locations,
                additional_attr,
            ) = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                self.base_url,
                self.generator_cfg,
                "swe-gym",
                sampling_params,
                max_tokens,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        except Exception as e:
            logger.error(f"Critical Error in conversation: {str(e)}", exc_info=True)
            # TODO properly handle this
            error = str(e) + "\n" + traceback.format_exc()
            messages = []
            sft_messages = []
            tool_schemas = []
            final_message = ""
            structured_locations = None
            additional_attr = {
                "wall_clock_duration": 0.0,
                "start_timestamp": None,
                "end_timestamp": None,
            }

        # Run sanity check before computing the reward so that the logged metrics reflect the actual reward received in training
        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        trajectory_exhausted_steps = (
            structured_locations is None
            and len(token_messages) >= self.generator_cfg.max_turns
        )

        # NOTE: The agent called the custom finish tool but there were some sanity check issues like calling the tool multiple times, having extra text after ending the tool-call, calling this tool in parallel with other tools etc. Give 0 reward in such cases.
        # NOTE: Similar checks are not done for previous turns
        if (
            structured_locations is not None
            and self.sanity_check_last_step(token_messages) == False
        ):
            # If sanity check fails, set structured_locations to None so that reward fns that depend on it give 0 reward
            structured_locations = None
            final_message = ""

        # Reward Manager
        reward = 0
        reward_dict = {}

        for reward_fn_args in self.generator_cfg.reward:
            try:
                input_args = {
                    "final_message": final_message,
                    "messages": messages,
                    "instance": instance,
                    "structured_locations": structured_locations,
                }

                reward_fn = get_reward_function(reward_fn_args["fn"])

                input_args = {**input_args, **reward_fn_args.get("args", {})}

                reward_weight = reward_fn_args.get("weight", 1.0)
                reward_outputs = reward_fn(**input_args)
                if isinstance(reward_outputs, tuple):
                    reward_value, reward_items = reward_outputs
                else:
                    reward_value = reward_outputs
                    reward_items = {reward_fn_args["fn"]: reward_value}
                reward_value = reward_value * reward_weight
            except Exception as e:
                logger.error(
                    f"Error in computing reward {reward_fn_args['fn']}: {e}",
                    exc_info=True,
                )
                reward_value = 0.0
                reward_items = {reward_fn_args["fn"]: reward_value}

            reward += reward_value

            reward_dict = {
                **reward_dict,
                **reward_items,
            }

        # Compute Trajectory Metrics
        efficiency_metrics = compute_all_efficiency_metrics(
            messages=messages,
            **additional_attr,
        )

        trajectory_metrics = compute_trajectory_metrics(messages)

        metrics_dict = {**efficiency_metrics, **trajectory_metrics}

        print(
            f"Total reward: {reward}\nReward details: {reward_dict}\nTrajectory metrics: {metrics_dict}"
        )

        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        rollout_list = []
        if len(token_messages) > 0:
            if self.step_wise:
                for idx, message in enumerate(token_messages):
                    current_prompt_ids = message["prompt_token_ids"]
                    current_response_ids = message["response_token_ids"]

                    rollout_list.append(
                        (
                            current_response_ids,
                            reward,
                            "complete",
                            [1] * len(current_response_ids),
                            current_prompt_ids,
                            None,
                            trajectory_metrics,
                        )
                    )
            else:
                try:
                    current_prompt_ids, current_response_ids, mask = (
                        build_assistant_loss_mask(token_messages)
                    )
                except ValueError as exc:
                    logger.error(f"Invalid TokenEvent prefix chain: {exc}")
                    current_prompt_ids = token_messages[0]["prompt_token_ids"]
                    current_response_ids = token_messages[-1]["response_token_ids"]
                    mask = [0] * len(current_response_ids)

                max_response_len = max(
                    0, self.max_train_length - len(current_prompt_ids)
                )
                if len(current_response_ids) > max_response_len:
                    current_response_ids = current_response_ids[:max_response_len]
                    mask = mask[:max_response_len]

                # mask loss completely from trajectories that exhausted all steps without calling the custom finish tool
                if trajectory_exhausted_steps:
                    logger.info(
                        "Trajectory exhausted all steps without calling the custom finish tool. Masking out loss from this rollout."
                    )
                    for i in range(len(mask)):
                        mask[i] = 0

                rollout_list.append(
                    (
                        current_response_ids,
                        reward,
                        "complete",
                        mask,
                        current_prompt_ids,
                        None,
                        trajectory_metrics,
                    )
                )

        else:
            # Ideally the code should not reach here
            logger.info(
                "IMPORTANT_ERROR: No TokenEvents found in the conversation. Saving an error rollout with minimal data."
            )
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [0]  # NOTE: Mask out loss completely
            initial_input_ids = [151643]
            trajectory_metrics = {}  # Empty metrics for error case
            rollout_list.append(
                (
                    response_ids,
                    reward,
                    stop_reason,
                    loss_mask,
                    initial_input_ids,
                    None,
                    trajectory_metrics,
                )
            )

        # Add "/" at the end of traj_dir if not present
        if not self.generator_cfg.traj_dir.endswith("/"):
            self.generator_cfg.traj_dir += "/"

        path = (
            self.generator_cfg.traj_dir
            + f"step_{batch_metadata.global_step}/{batch_metadata.training_phase}/"
        )
        # Check if traj_dir is a gcs path
        if path.startswith("gs://"):
            use_gcs = True
            fs = gcsfs.GCSFileSystem()
        else:
            use_gcs = False
            fs = fsspec.filesystem("file")
            # Pre-create directory to avoid race conditions with parallel workers
            os.makedirs(path, exist_ok=True)

        instance_id = env_extras["instance_id"]

        if error is not None:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.error"
            filename_path = path + filename
            print(f"Saving error to {filename_path}")
            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                f.write(error)
        else:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
            filename_path = path + filename

            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)

            # get everything between ```` with regex
            try:
                raw_final_message = (
                    json.dumps(structured_locations)
                    if structured_locations is not None
                    else final_message
                )
            except Exception as e:
                raw_final_message = ""
            matches = re.findall(r"```(.*?)```", final_message, re.DOTALL)
            parsed_final_message = matches[-1] if matches else final_message

            # Force messages to be JSON serializable
            for msg in messages:
                for key, value in msg.items():
                    try:
                        json.dumps(value)
                    except (TypeError, OverflowError):
                        msg[key] = str(value)

            result_dict = {
                "instance_id": instance_id,
                "target": env_extras["target"],
                "total_reward": reward,
                "reward_dict": reward_dict,
                "parsed_final_message": parsed_final_message,
                "raw_final_message": raw_final_message,
                "messages": messages,
                "sft_messages": sft_messages,
                "tools": tool_schemas,
                "metrics_dict": metrics_dict,
            }

            print(f"Saving trajectory to {filename_path}")
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                json.dump(
                    result_dict, f, indent=2
                )  # , sort_keys=True, ensure_ascii=False)

        return [rollout_list, reward_dict, metrics_dict]

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids")
        batch_metadata = input_batch.get("batch_metadata")
        if trajectory_ids is None or batch_metadata is None:
            raise ValueError("trajectory_ids and batch_metadata are required")
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = input_batch.get("sampling_params")
        if sampling_params is None:
            sampling_params = asdict(self.generator_cfg.sampling_params)

        task_rollouts = []
        for i in range(len(prompts)):
            rollout = self.code_search_loop(
                prompts[i],
                env_extras[i],
                max_tokens=max_tokens,
                max_input_length=max_input_length,
                sampling_params=sampling_params,
                trajectory_id=trajectory_ids[i],
                batch_metadata=batch_metadata,
            )

            task_rollouts.append(rollout)

        collected_task_rollouts = await asyncio.gather(*task_rollouts)

        all_outputs = [rollout[0] for rollout in collected_task_rollouts]
        rewards_dict = [rollout[1] for rollout in collected_task_rollouts]
        metrics_dict = [rollout[2] for rollout in collected_task_rollouts]

        responses = sum(
            [[output[0] for output in step_outputs] for step_outputs in all_outputs], []
        )
        rewards = sum(
            [[output[1] for output in step_outputs] for step_outputs in all_outputs], []
        )
        stop_reasons = sum(
            [[output[2] for output in step_outputs] for step_outputs in all_outputs], []
        )
        loss_masks = sum(
            [[output[3] for output in step_outputs] for step_outputs in all_outputs], []
        )
        prompt_token_ids = sum(
            [[output[4] for output in step_outputs] for step_outputs in all_outputs], []
        )

        out_trajectory_ids = []
        is_last_step = []
        for i, step_outputs in enumerate(all_outputs):
            for step_id in range(len(step_outputs)):
                out_trajectory_ids.append(trajectory_ids[i])
                is_last_step.append(step_id == len(step_outputs) - 1)

        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards, loss_masks=loss_masks)

        tracked_metrics = {}

        # Aggregate Rewards and Metrics
        for tracker_name, tracker_dict in zip(
            ["reward", "metrics"], [rewards_dict, metrics_dict]
        ):
            for tracker_dict_item in tracker_dict:
                for k, v in tracker_dict_item.items():
                    # Check if v is numeric
                    if not isinstance(v, (int, float)):
                        continue
                    if f"{tracker_name}/{k}" not in tracked_metrics:
                        tracked_metrics[f"{tracker_name}/{k}"] = []
                    tracked_metrics[f"{tracker_name}/{k}"].append(v)

        # Average all tracked metrics
        for k, v in tracked_metrics.items():
            tracked_metrics[k] = sum(v) / len(v)

        generator_output: GeneratorOutput = {
            "trajectory_ids": out_trajectory_ids,
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "trajectory_generation_times": [
                float(item.get("wall_clock_duration", 0.0)) for item in metrics_dict
            ],
            "rollout_expert_indices": None,
            "is_last_step": is_last_step if self.step_wise else None,
            "env_metrics": sum(
                [
                    ([metric] * len(outputs))
                    for metric, outputs in zip(metrics_dict, all_outputs)
                ],
                [],
            ),
            "pixel_values": None,
            "image_grid_thw": None,
            **tracked_metrics,
        }

        return generator_output
