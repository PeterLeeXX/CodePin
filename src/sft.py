"""Qwen3.5 language-model-only SFT entrypoint for SkyRL v0.3."""

from __future__ import annotations

import sys

import ray
from skyrl.train.config.sft_config import (
    SFTConfig,
    build_skyrl_config_for_sft,
    validate_sft_cfg,
)
from skyrl.train.main_sft import sft_entrypoint
from skyrl.train.utils.utils import initialize_ray


def main() -> None:
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    validate_sft_cfg(cfg)
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    # CodePin is text-only; do not load Qwen3.5's unused vision tower.
    skyrl_cfg.trainer.policy.language_model_only = True
    initialize_ray(skyrl_cfg)
    ray.get(sft_entrypoint.remote(cfg, skyrl_cfg))


if __name__ == "__main__":
    main()
