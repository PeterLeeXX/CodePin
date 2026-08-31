"""CodePin RL entrypoint for the typed SkyRL v0.3 configuration API."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ray
import torch
import yaml
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import BaseConfig, GeneratorConfig, SkyRLTrainConfig
from skyrl.train.config.config import build_nested_dataclass
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from src.generator.code_search_generator import CodeSearchGenerator


@dataclass
class PromptPaths(BaseConfig):
    system_prompt: str = "templates/system_prompt_atomic_search.j2"
    user_prompt: str = "templates/file_module_custom_finish.j2"


@dataclass
class CodeSearchGeneratorConfig(GeneratorConfig):
    reward: list[dict[str, Any]] = field(
        default_factory=lambda: [{"fn": "multilevel_localization_f1_reward"}]
    )
    tools: list[str] = field(
        default_factory=lambda: ["glob", "grep", "read_file"]
    )
    prompts: PromptPaths = field(default_factory=PromptPaths)
    traj_dir: str = "ckpts/trajectories"
    max_train_length: int = 40960
    exp_config: Optional[str] = None


@dataclass
class OPDConfig(BaseConfig):
    enabled: bool = False
    teacher_model: Optional[str] = None
    distill_coef: float = 1.0
    task_reward_coef: float = 1.0
    reward_clip: float = 5.0


@dataclass
class CodeSearchSkyRLConfig(SkyRLTrainConfig):
    generator: CodeSearchGeneratorConfig = field(
        default_factory=CodeSearchGeneratorConfig
    )
    opd: OPDConfig = field(default_factory=OPDConfig)

    def __post_init__(self) -> None:
        if isinstance(self.generator, dict):
            self.generator = build_nested_dataclass(CodeSearchGeneratorConfig, self.generator)
        if isinstance(self.opd, dict):
            self.opd = OPDConfig(**self.opd)
        super().__post_init__()


def normalize_opd_config(cfg: CodeSearchSkyRLConfig) -> OPDConfig:
    if isinstance(cfg.opd, OPDConfig):
        return cfg.opd
    cfg.opd = OPDConfig(**cfg.opd)
    return cfg.opd


class CodeSearchOPDTrainer(RayPPOTrainer):
    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Replace the reference KL penalty with reverse-KL OPD token rewards."""
        loss_mask: torch.Tensor = data["loss_mask"]
        task_rewards: torch.Tensor = data["rewards"]
        teacher_log_probs: Optional[torch.Tensor] = data["base_action_log_probs"]
        action_log_probs: Optional[torch.Tensor] = data["action_log_probs"]

        if teacher_log_probs is None:
            raise RuntimeError("OPD requires trainer.ref.model.path to load a teacher model.")
        if action_log_probs is None:
            raise RuntimeError("OPD requires policy logprobs; use a policy loss that keeps policy forward enabled.")

        mask = loss_mask.to(action_log_probs.dtype)
        opd_rewards = (teacher_log_probs - action_log_probs) * mask
        opd_cfg = normalize_opd_config(self.cfg)
        if opd_cfg.reward_clip > 0:
            opd_rewards = torch.clamp(
                opd_rewards,
                min=-opd_cfg.reward_clip,
                max=opd_cfg.reward_clip,
            )

        data["rewards"] = (
            opd_cfg.distill_coef * opd_rewards
            + opd_cfg.task_reward_coef * task_rewards
        )

        denom = mask.sum().clamp_min(1.0)
        avg_opd_reward = (opd_rewards * mask).sum() / denom
        avg_task_reward = task_rewards.sum(dim=-1).mean()
        avg_total_reward = data["rewards"].sum(dim=-1).mean()
        valid_opd = opd_rewards[mask > 0]
        max_abs_opd_reward = (
            valid_opd.abs().max() if valid_opd.numel() else avg_opd_reward.new_tensor(0.0)
        )

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        metrics = {
            "opd/avg_token_reward": avg_opd_reward.item(),
            "opd/max_abs_token_reward": max_abs_opd_reward.item(),
            "opd/avg_task_reward": avg_task_reward.item(),
            "opd/avg_total_reward": avg_total_reward.item(),
            "opd/distill_coef": opd_cfg.distill_coef,
            "opd/task_reward_coef": opd_cfg.task_reward_coef,
            "opd/reward_clip": opd_cfg.reward_clip,
        }
        data.metadata["metrics"].update(metrics)
        self.all_metrics.update(metrics)

        return data


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return CodeSearchGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        trainer_cls = CodeSearchOPDTrainer if normalize_opd_config(cfg).enabled else RayPPOTrainer
        return trainer_cls(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


def apply_experiment_config(cfg: CodeSearchSkyRLConfig) -> None:
    if not cfg.generator.exp_config:
        return
    path = Path(cfg.generator.exp_config)
    with path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle) or {}
    for key in ("reward", "tools"):
        if key in experiment:
            setattr(cfg.generator, key, experiment[key])
    if "prompts" in experiment:
        cfg.generator.prompts = PromptPaths(**experiment["prompts"])


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: CodeSearchSkyRLConfig) -> None:
    CodeSearchPPOExp(cfg).run()


def main() -> None:
    cfg = CodeSearchSkyRLConfig.from_cli_overrides(sys.argv[1:])
    apply_experiment_config(cfg)
    opd_cfg = normalize_opd_config(cfg)
    if opd_cfg.enabled:
        if opd_cfg.teacher_model:
            cfg.trainer.ref.model.path = opd_cfg.teacher_model
        if not cfg.trainer.ref.model.path:
            raise ValueError("OPD requires opd.teacher_model or trainer.ref.model.path.")
        cfg.trainer.algorithm.use_kl_in_reward = True
        cfg.trainer.algorithm.use_kl_loss = False
    if cfg.trainer.fully_async.enabled:
        raise ValueError(
            "CodePin's OpenHands HTTP loop does not yet return rollout token "
            "logprobs; use synchronous on-policy mode (fully_async.enabled=false)."
        )
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
