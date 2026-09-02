<div align="center">
  <img src="docs/codepin.svg" alt="CodePin logo" width="140" />
  <h1>CodePin</h1>
  <p><strong>Pin the right code before you write the code.</strong></p>
</div>

CodePin is a small, read-only code-localization agent. Given a Python issue and
a repository snapshot, it searches with `glob`, `grep`, and `read_file`, then
returns the files, classes, and functions that need modification through one
structured `localization_finish` call.

## Scope

CodePin contains one production path:

```text
SWE-Smith data -> Qwen3.5 SFT -> synchronous DAPO/GRPO -> structured locations
```

The agent does not edit code, execute shell commands in target repositories, or
generate patches. The fixed read-only action space keeps trajectories bounded,
reproducible, and inexpensive.

## Environment

CodePin intentionally has no runtime compatibility layer. The supported host is:

- Linux x86_64
- Python 3.12
- NVIDIA Ampere or newer with CUDA and bf16
- `ripgrep`

Install and validate the exact environment before starting a job:

```bash
sudo apt-get update && sudo apt-get install -y ripgrep
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
uv run python scripts/preflight_qwen3_5.py
```

Unsupported hosts fail during installation or preflight instead of being patched
inside the application.

## Data

The repository keeps only a 100-row validation sample under `data/sample`.
Download full datasets when training:

```bash
huggingface-cli download LeeXugar/SWE-smith-code-search \
  --repo-type dataset --local-dir data/SWE-smith-code-search

huggingface-cli download LeeXugar/CodePin-SFT-Qwen3.5-35B-A3B \
  --repo-type dataset --local-dir data/CodePin-SFT-Qwen3.5-35B-A3B
```

Raw SWE-Smith shards can also be converted with the retained AST-based builder:

```bash
uv run python -m src.build_swe_smith_code_search \
  --input /path/to/SWE-smith-shards \
  --output data/SWE-smith-code-search \
  --overwrite
```

## Training

Full-parameter SFT uses the released tool-call trajectories:

```bash
DATA_DIR=data/CodePin-SFT-Qwen3.5-35B-A3B/data \
  bash scripts/run_sft_qwen3_5_0_8b.sh
```

Continue from the exported SFT model with synchronous on-policy RL:

```bash
MODEL=/absolute/path/to/sft/hf_export \
DATA_PATH=data/SWE-smith-code-search \
  bash scripts/run_rl_qwen3_5_0_8b.sh
```

For native SkyRL DAPO (including dynamic sampling and overlong filtering), use:

```bash
MODEL=/absolute/path/to/sft/hf_export \
DATA_PATH=data/SWE-smith-code-search \
  bash scripts/run_dapo_qwen3_5_0_8b.sh
```

The main implementation is in `src/generator/code_search_generator.py`; data
labeling, tools, reward, SFT, and RL entrypoints each have one corresponding
module under `src/`.

## Released artifacts

- [CodePin-SFT-Qwen3.5-0.8B](https://huggingface.co/LeeXugar/CodePin-SFT-Qwen3.5-0.8B)
- [SWE-smith Code Search](https://huggingface.co/datasets/LeeXugar/SWE-smith-code-search)
- [CodePin SFT trajectories](https://huggingface.co/datasets/LeeXugar/CodePin-SFT-Qwen3.5-35B-A3B)

CodePin is released under the MIT License.
