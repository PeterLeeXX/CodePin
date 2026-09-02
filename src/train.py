"""CodePin synchronous GRPO/GSPO training entrypoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import ray
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig
from skyrl.train.config.config import build_nested_dataclass
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from src.generator.code_search_generator import CodeSearchGenerator


@dataclass
class CodeSearchGeneratorConfig(GeneratorConfig):
    traj_dir: str = "ckpts/trajectories"
    max_train_length: int = 40960
    efficiency_weight: float = 0.2
    result_cache: bool = False


@dataclass
class CodeSearchSkyRLConfig(SkyRLTrainConfig):
    generator: CodeSearchGeneratorConfig = field(
        default_factory=CodeSearchGeneratorConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.generator, dict):
            self.generator = build_nested_dataclass(
                CodeSearchGeneratorConfig, self.generator
            )
        super().__post_init__()


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return CodeSearchGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            policy_model_name=resolve_policy_model_name(cfg),
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: CodeSearchSkyRLConfig) -> None:
    CodeSearchPPOExp(cfg).run()


def main() -> None:
    cfg = CodeSearchSkyRLConfig.from_cli_overrides(sys.argv[1:])
    if cfg.trainer.fully_async.enabled:
        raise ValueError("CodePin supports synchronous on-policy training only")
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
