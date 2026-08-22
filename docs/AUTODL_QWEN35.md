# AutoDL：Qwen3.5-0.8B SFT → RL

本仓库固定使用经过 SkyRL v0.3 验证的组合：Python 3.12、Torch 2.11、
Transformers 5.6.1–5.8.0、vLLM 0.23.0。Qwen3.5 的训练端、参考模型与推理端均启用
`language_model_only`；FSDP 的 `remove_microbatch_padding` 关闭，工具解析器使用
`qwen3_coder`，对话生成关闭 thinking。

## 1. AutoDL 初始化

建议选择至少 24 GB 显存的 Ampere 或更新 GPU。长上下文的主要显存压力来自激活与
logits，而不是 0.8B 参数本身；32K SFT 推荐 48 GB，24 GB 先把 `MAX_LENGTH` 降到
8192 或 16384。

```bash
cd /root/CodePin
apt-get update && apt-get install -y ripgrep
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv sync
uv run python scripts/preflight_qwen3_5.py
```

预检必须显示 CUDA、bf16、`qwen3_5` model type，并能导入 `causal_conv1d` 与
`flash_linear_attention`。失败时不要开始长任务。

## 2. 准备监督轨迹

SFT 数据应来自成功的工具调用轨迹，而不是 RL 的训练/验证样本答案。每条记录使用
OpenAI `messages` 格式并保留 `tools` schema；默认只选总奖励不低于 1 的轨迹，去重后
按固定种子划分训练集和验证集。工具 schema 必须严格按
`glob`、`grep`、`read_file`、`localization_finish` 排列；旧 `terminal` 轨迹会被过滤，
避免混入不兼容的动作空间。

```bash
uv run python scripts/prepare_sft_data.py \
  --trajectories /root/autodl-tmp/teacher-trajectories \
  --output /root/autodl-tmp/codepin-sft \
  --exclude-instance-ids /root/autodl-tmp/rl-eval-instance-ids.txt
```

`--exclude-instance-ids` 用来避免 SFT 与后续 RL/eval 泄漏。脚本输出
数据目录中的 `train.jsonl`、`validation.jsonl`，以及同级的
`codepin-sft-report.json`。新 CodePin rollout 会自动保存可直接使用的
`sft_messages` 与 `tools` 字段。

## 3. SFT

默认执行全参数 SFT，而不是 LoRA：0.8B 模型可以承受，且与 CodePin 原始训练设定一致。
配置为 1 epoch、全局 batch 8、micro batch 1、AdamW、学习率 `5e-5`、cosine 调度，
训练所有 assistant 轮次并屏蔽 user/tool observation。

```bash
DATA_DIR=/root/autodl-tmp/codepin-sft \
OUTPUT_DIR=/root/autodl-tmp/qwen35-08b-sft \
LOGGER=console \
MAX_LENGTH=32768 \
  bash scripts/run_sft_qwen3_5_0_8b.sh
```

如果训练样本数不是约 4000 条，应按 `ceil(样本数 / BATCH_SIZE * 0.1)` 设置
`WARMUP_STEPS`。首次运行建议在命令末尾追加 `max_training_steps=2` 做冒烟测试，确认
loss、显存和 HF 导出正常后再跑完整任务。

## 4. RL

将 `MODEL` 指向 SFT 生成的 HF export，而不是 SkyRL 的分布式 checkpoint 目录：

```bash
MODEL=/root/autodl-tmp/qwen35-08b-sft/hf_exports/global_step_500 \
DATA_PATH=/root/CodePin/data/SWE-smith-code-search \
OUTPUT_DIR=/root/autodl-tmp/qwen35-08b-rl \
LOGGER=console \
  bash scripts/run_rl_qwen3_5_0_8b.sh
```

当前 CodePin/OpenHands 自定义工具循环采用同步 on-policy GRPO/GSPO。完整多轮 HTTP 工具
轨迹尚未回传 rollout token logprobs；同步模式会在训练端精确重算 old logprobs，避免把
旧版“异步但无 rollout logprobs”的路径错误迁移到 SkyRL v0.3。旧的三个
`run_async_training_*` 文件仅作为兼容入口，均转发到该 Qwen3.5 配方。

单卡冒烟测试可缩小为：

```bash
NUM_GPUS=1 NUM_ENGINES=1 N_ROLLOUTS=2 BATCH_SIZE=2 \
MAX_PROMPT_LENGTH=8192 MAX_GENERATE_LENGTH=2048 MAX_TRAINING_STEPS=2 \
MODEL=/absolute/path/to/sft-export \
  bash scripts/run_rl_qwen3_5_0_8b.sh
```

训练日志与 checkpoint 放在 `/root/autodl-tmp`，避免系统盘爆满。正式运行建议在 tmux 中
启动，并保留 `report.json`、完整命令、GPU 型号、git commit、日志和最终 HF export。
