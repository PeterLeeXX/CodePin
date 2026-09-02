"""Wrap the released text checkpoint in the native Qwen3.5 HF serving format.

Weights are hard-linked without changing a byte. vLLM's built-in conditional
generation class understands the released model.language_model.* weight names;
--language-model-only suppresses the absent vision tower. No runtime patching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer, Qwen3_5Config


def prepare(source: Path, output: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_text":
        raise ValueError("expected the released qwen3_5_text SFT checkpoint")
    if not (source / "model.safetensors").is_file():
        raise ValueError("expected model.safetensors from the released SFT model")
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.eos_token_id is None:
        raise ValueError("the serving tokenizer must declare a chat EOS token")
    original_generation = json.loads((source / "generation_config.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    wrapper = Qwen3_5Config(text_config=config)
    wrapper.text_config.eos_token_id = tokenizer.eos_token_id
    wrapper.architectures = ["Qwen3_5ForConditionalGeneration"]
    wrapper.save_pretrained(output)
    hashes = {}
    for name in (
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = source / name
        # Source and output must live on the same filesystem, as documented.
        os.link(path, output / name)
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    # The release declares endoftext as generation EOS, but its chat tokenizer
    # declares im_end. A second, unrecognized stop token can terminate a JSON
    # grammar before the object closes. Use the tokenizer's native chat EOS.
    generation = {**original_generation, "eos_token_id": tokenizer.eos_token_id}
    (output / "generation_config.json").write_text(json.dumps(generation, indent=2))
    manifest = {
        "format": "native_qwen3_5_language_model_only",
        "weights_unchanged": True,
        "source": str(source.resolve()),
        "sha256": hashes,
        "source_config": config,
        "source_generation_config": original_generation,
        "serving_generation_config": generation,
    }
    (output / "codepin_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
