"""Fail fast unless the host matches CodePin's supported runtime."""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required, got {platform.python_version()}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("CodePin supports Linux x86_64 only")

    rg = shutil.which("rg")
    if rg is None:
        raise RuntimeError("ripgrep is required")
    rg_version = subprocess.run(
        [rg, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]

    import torch
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CodePin requires a bf16-capable NVIDIA GPU")
    unsupported = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index)[0] < 8
    ]
    if unsupported:
        raise RuntimeError(f"Ampere or newer GPUs are required: {unsupported}")

    for module in (
        "skyrl",
        "vllm",
        "flash_attn",
        "causal_conv1d",
        "fla",
    ):
        importlib.import_module(module)

    config = AutoConfig.from_pretrained(args.model)
    if "qwen3_5" not in str(getattr(config, "model_type", "")):
        raise RuntimeError(f"Unexpected model type: {config.model_type!r}")

    print(f"python={platform.python_version()}")
    print(f"ripgrep={rg_version}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformers={version('transformers')}")
    print(f"vllm={version('vllm')}")
    print(f"skyrl={version('skyrl')}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
    print("CodePin preflight: OK")


if __name__ == "__main__":
    main()
