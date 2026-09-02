#!/usr/bin/env bash
set -euo pipefail

# AutoDL's conda runtime may shadow the newer system libstdc++ required by
# vllm-router's optional Rust extension.
if [[ -z "${LD_PRELOAD:-}" && -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi

EXTRA_OVERRIDES=()
RESUME_PATH=""
RUN_NAME="${RUN_NAME:-qwen3.5-0.8b-dapo}"
while getopts ":m:n:d:s:l:o:i:t:b:c:r:" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    n) N_ROLLOUTS="$OPTARG" ;;
    d) DATA_PATH="$OPTARG" ;;
    s) OUTPUT_DIR="$OPTARG" ;;
    l) RESUME_PATH="$OPTARG" ;;
    o) EXTRA_OVERRIDES+=("$OPTARG") ;;
    i) NUM_ENGINES="$OPTARG" ;;
    t) NUM_GPUS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    c) MICRO_BATCH_SIZE="$OPTARG" ;;
    r) RUN_NAME="$OPTARG" ;;
    *) echo "Unknown legacy option: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

MODEL="${MODEL:-LeeXugar/CodePin-SFT-Qwen3.5-0.8B}"
DATA_PATH="${DATA_PATH:-data/SWE-smith-code-search}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/Qwen-Qwen3.5-0.8B-dapo}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
NUM_ENGINES="${NUM_ENGINES:-$NUM_GPUS}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MICRO_FORWARD_BATCH_SIZE="${MICRO_FORWARD_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
MAX_ENGINE_LENGTH="${MAX_ENGINE_LENGTH:-40960}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-131072}"
FLASH_ATTN="${FLASH_ATTN:-false}"
REMOVE_MICROBATCH_PADDING="${REMOVE_MICROBATCH_PADDING:-false}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-16}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-100}"
LOGGER="${LOGGER:-console}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "$RESUME_PATH" ]]; then
  EXTRA_OVERRIDES+=("trainer.resume_mode=from_path" "trainer.resume_path=$RESUME_PATH")
fi

mkdir -p "$OUTPUT_DIR"

# SkyRL supplies the native DAPO path: GRPO advantages, clip-higher, dynamic
# filtering, token-level loss, and overlong filtering. MODEL is the existing
# SFT export; tokenizer/model settings remain inherited from that checkpoint.
if [[ -n "$PYTHON_BIN" ]]; then
  TRAIN_CMD=("$PYTHON_BIN" -m src.train)
else
  TRAIN_CMD=(uv run --isolated -m src.train)
fi

"${TRAIN_CMD[@]}" \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.policy_loss_type=dual_clip \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.dynamic_sampling.type=filter \
  trainer.algorithm.dynamic_sampling.max_sample_batches=30 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model_config_kwargs.rope_parameters='{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}' \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.flash_attn="$FLASH_ATTN" \
  trainer.remove_microbatch_padding="$REMOVE_MICROBATCH_PADDING" \
  trainer.epochs=1 \
  trainer.max_training_steps="$MAX_TRAINING_STEPS" \
  trainer.train_batch_size="$BATCH_SIZE" \
  trainer.policy_mini_batch_size="$BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD_BATCH_SIZE" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  trainer.max_prompt_length="$MAX_PROMPT_LENGTH" \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.ckpt_path="$OUTPUT_DIR/checkpoints" \
  trainer.export_path="$OUTPUT_DIR/hf_exports" \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=50 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=null \
  trainer.logger="$LOGGER" \
  trainer.project_name=codepin-dapo \
  trainer.run_name="$RUN_NAME" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_ENGINE_LENGTH" \
  generator.inference_engine.engine_init_kwargs.block_size=16 \
  generator.inference_engine.engine_init_kwargs.mamba_block_size="$MAMBA_BLOCK_SIZE" \
  generator.inference_engine.engine_init_kwargs.mamba_cache_mode=align \
  generator.inference_engine.engine_init_kwargs.hf_overrides='{"rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}' \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=qwen3_coder \
  generator.batched=false \
  generator.n_samples_per_prompt="$N_ROLLOUTS" \
  generator.max_turns=6 \
  generator.max_input_length="$MAX_PROMPT_LENGTH" \
  generator.max_train_length="$MAX_ENGINE_LENGTH" \
  generator.apply_overlong_filtering=true \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=20 \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.traj_dir="$OUTPUT_DIR/trajectories" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
