"""Qwen3.5 language-model-only SFT entrypoint for SkyRL v0.3."""

from __future__ import annotations

import os
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
    # Qwen3.5 is packaged as a conditional-generation model. CodePin is text
    # only, so avoid loading the unused vision tower during SFT.
    skyrl_cfg.trainer.policy.language_model_only = True
    # V100 (sm70) cannot use the pinned FlashAttention/PyTorch 2.11 kernels.
    # The official Transformers Qwen3.5 implementation has SDPA and pure-Torch
    # Gated DeltaNet fallbacks, selected explicitly for the compatibility run.
    if os.environ.get("CODEPIN_DISABLE_FLASH_ATTN", "0") == "1":
        skyrl_cfg.trainer.flash_attn = False
    initialize_ray(skyrl_cfg)
    ray.get(sft_entrypoint.remote(cfg, skyrl_cfg))


if __name__ == "__main__":
    main()
