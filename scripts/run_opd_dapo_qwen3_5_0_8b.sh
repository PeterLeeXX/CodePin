#!/usr/bin/env bash
set -euo pipefail

EXTRA_OVERRIDES=()
RESUME_PATH=""
STEP_WISE="${STEP_WISE:-false}"
RUN_NAME="${RUN_NAME:-qwen3.5-0.8b-opd-dapo}"
while getopts ":m:e:n:d:s:l:o:i:t:b:c:r:w:" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    e) TEACHER_MODEL="$OPTARG" ;;
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
    w) STEP_WISE="$OPTARG" ;;
    *) echo "Unknown legacy option: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

: "${TEACHER_MODEL:?Set TEACHER_MODEL to a same-tokenizer teacher model path or HF repo.}"

MODEL="${MODEL:-LeeXugar/CodePin-SFT-Qwen3.5-0.8B}"
DATA_PATH="${DATA_PATH:-data/SWE-smith-code-search}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/Qwen-Qwen3.5-0.8B-opd-dapo}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
NUM_ENGINES="${NUM_ENGINES:-$NUM_GPUS}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-32768}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-8192}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-100}"
LOGGER="${LOGGER:-console}"
OPD_DISTILL_COEF="${OPD_DISTILL_COEF:-1.0}"
OPD_TASK_REWARD_COEF="${OPD_TASK_REWARD_COEF:-1.0}"
OPD_REWARD_CLIP="${OPD_REWARD_CLIP:-5.0}"
TRAINER_BF16="${TRAINER_BF16:-false}"
VLLM_DTYPE="${VLLM_DTYPE:-half}"

if [[ -n "$RESUME_PATH" ]]; then
  EXTRA_OVERRIDES+=("trainer.resume_mode=from_path" "trainer.resume_path=$RESUME_PATH")
fi

mkdir -p "$OUTPUT_DIR"

export PATH="${CODEPIN_EXTRA_PATH:-/root/.local/bin:/root/miniconda3/bin}${PATH:+:${PATH}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(pwd)/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
if [[ -n "${CODEPIN_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${CODEPIN_EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ -n "${CODEPIN_COMPAT_PATH:-}" ]]; then
  export PYTHONPATH="${CODEPIN_COMPAT_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ -n "${CODEPIN_OPD_PYTHON:-}" ]]; then
  TRAIN_COMMAND=("$CODEPIN_OPD_PYTHON" -m src.train)
else
  TRAIN_COMMAND=(uv run -m src.train)
fi

"${TRAIN_COMMAND[@]}" \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  opd.enabled=true \
  opd.teacher_model="$TEACHER_MODEL" \
  opd.distill_coef="$OPD_DISTILL_COEF" \
  opd.task_reward_coef="$OPD_TASK_REWARD_COEF" \
  opd.reward_clip="$OPD_REWARD_CLIP" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.policy_loss_type=gspo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.eps_clip_low=0.0003 \
  trainer.algorithm.eps_clip_high=0.0004 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.algorithm.dynamic_sampling.type=filter_zero_variance \
  trainer.algorithm.dynamic_sampling.max_sample_batches=4 \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  trainer.policy.optimizer_config.lr=5e-7 \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.remove_microbatch_padding=false \
  trainer.flash_attn=false \
  trainer.bf16="$TRAINER_BF16" \
  trainer.epochs=1 \
  trainer.max_training_steps="$MAX_TRAINING_STEPS" \
  trainer.train_batch_size="$BATCH_SIZE" \
  trainer.policy_mini_batch_size="$BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu=1 \
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
  trainer.project_name=codepin-opd-dapo \
  trainer.run_name="$RUN_NAME" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.model_dtype="$VLLM_DTYPE" \
  generator.inference_engine.gpu_memory_utilization=0.72 \
  generator.inference_engine.max_num_batched_tokens=65536 \
  generator.inference_engine.engine_init_kwargs.max_model_len=40960 \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=qwen3_coder \
  generator.batched=false \
  generator.n_samples_per_prompt="$N_ROLLOUTS" \
  generator.max_turns=6 \
  generator.max_input_length="$MAX_PROMPT_LENGTH" \
  generator.max_train_length=40960 \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=20 \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.step_wise_trajectories="$STEP_WISE" \
  generator.traj_dir="$OUTPUT_DIR/trajectories" \
  generator.exp_config=configs/reward_config_1.7b.yaml \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
