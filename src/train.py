"""CodePin RL entrypoint for the typed SkyRL v0.3 configuration API."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ray
import yaml
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import BaseConfig, GeneratorConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
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
class CodeSearchSkyRLConfig(SkyRLTrainConfig):
    generator: CodeSearchGeneratorConfig = field(
        default_factory=CodeSearchGeneratorConfig
    )


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return CodeSearchGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
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
