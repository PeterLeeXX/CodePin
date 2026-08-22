#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
DATA_DIR="${DATA_DIR:-data/sft-code-search}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/Qwen-Qwen3.5-0.8B-sft}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
LOGGER="${LOGGER:-console}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"

mkdir -p "$OUTPUT_DIR"

# Qwen3.5 FSDP currently requires padding removal / sample packing to stay off.
uv run --isolated -m src.sft \
  strategy=fsdp \
  model.path="$MODEL" \
  dataset_name="$DATA_DIR" \
  dataset_split=train \
  eval_dataset_name="$DATA_DIR" \
  eval_dataset_split=validation \
  eval_interval=0 \
  messages_key=messages \
  tools_key=tools \
  train_on_what=all_assistant_messages \
  max_length="$MAX_LENGTH" \
  num_epochs=1 \
  placement.num_nodes=1 \
  placement.num_gpus_per_node="$NUM_GPUS" \
  batch_size="$BATCH_SIZE" \
  micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  remove_microbatch_padding=false \
  fsdp_config.cpu_offload=false \
  fsdp_config.reshard_after_forward=true \
  optimizer_config.lr=5e-5 \
  optimizer_config.scheduler=cosine \
  optimizer_config.num_warmup_steps="$WARMUP_STEPS" \
  logger="$LOGGER" \
  project_name=codepin-sft \
  run_name=qwen3.5-0.8b-sft \
  ckpt_path="$OUTPUT_DIR/checkpoints" \
  ckpt_interval=100 \
  hf_save_interval=100 \
  export_path="$OUTPUT_DIR/hf_exports" \
  "$@"
