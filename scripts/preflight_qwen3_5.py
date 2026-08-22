#!/usr/bin/env python3
"""Fail-fast checks for the CodePin Qwen3.5 AutoDL environment."""

from __future__ import annotations

import importlib
import platform
import sys


def version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, got {platform.python_version()}")

    import torch
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bf16 training")

    importlib.import_module("skyrl")
    importlib.import_module("vllm")
    importlib.import_module("causal_conv1d")
    importlib.import_module("flash_linear_attention")

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    model_type = str(getattr(config, "model_type", ""))
    if "qwen3_5" not in model_type:
        raise RuntimeError(f"Unexpected model_type: {model_type!r}")

    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformers={version('transformers')}")
    print(f"vllm={version('vllm')}")
    print(f"skyrl={version('skyrl')}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
    print(f"model_type={model_type}")
    print("Qwen3.5 preflight: OK")


if __name__ == "__main__":
    main()
