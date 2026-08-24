#!/usr/bin/env python3
"""Fail-fast checks for the CodePin Qwen3.5 AutoDL environment."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys


def version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, got {platform.python_version()}")

    rg_executable = shutil.which("rg")
    if rg_executable is None:
        raise RuntimeError("ripgrep (rg) is required for CodePin atomic grep")
    rg_version = subprocess.run(
        [rg_executable, "--version"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()[0]

    import torch
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    precision = os.environ.get("CODEPIN_PRECISION", "bf16").lower()
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bf16 training; set CODEPIN_PRECISION=fp16 for V100")
    if precision not in {"bf16", "fp16"}:
        raise RuntimeError(f"Unsupported CODEPIN_PRECISION: {precision!r}")

    importlib.import_module("skyrl")
    importlib.import_module("vllm")
    importlib.import_module("causal_conv1d")
    # The PyPI distribution is named flash-linear-attention, while its Python
    # import package is `fla` in the Qwen3.5-compatible releases.
    importlib.import_module("fla")

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    model_type = str(getattr(config, "model_type", ""))
    if "qwen3_5" not in model_type:
        raise RuntimeError(f"Unexpected model_type: {model_type!r}")

    print(f"python={platform.python_version()}")
    print(f"ripgrep={rg_version}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"precision={precision}")
    print(f"transformers={version('transformers')}")
    print(f"vllm={version('vllm')}")
    print(f"skyrl={version('skyrl')}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
    print(f"model_type={model_type}")
    print("Qwen3.5 preflight: OK")


if __name__ == "__main__":
    main()
