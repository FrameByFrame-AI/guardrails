#!/usr/bin/env bash
set -euo pipefail

: "${PF_MODEL:=/models/privacy-filter}"
: "${PF_DATA_DIR:=/workspace/data/generated/ko_pii_opf_v4}"
: "${PF_OUTPUT_DIR:=/workspace/data/checkpoints/ko_pii_hf_ddp_v4}"
: "${PF_NPROC_PER_NODE:=2}"
: "${PF_EPOCHS:=5}"
: "${PF_BATCH_SIZE:=16}"
: "${PF_EVAL_BATCH_SIZE:=32}"
: "${PF_GRAD_ACCUM:=4}"
: "${PF_LEARNING_RATE:=2e-5}"
: "${PF_LR_SCHEDULER_TYPE:=cosine}"
: "${PF_WEIGHT_DECAY:=0.01}"
: "${PF_WARMUP_RATIO:=0.1}"
: "${PF_MAX_GRAD_NORM:=1.0}"
: "${PF_MAX_LENGTH:=512}"
: "${PF_DATALOADER_WORKERS:=4}"
: "${PF_LOGGING_STEPS:=25}"
: "${PF_SAVE_TOTAL_LIMIT:=2}"
: "${PF_SEED:=42}"
: "${PF_EARLY_STOPPING_PATIENCE:=2}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PF_RUN_ID="${PF_RUN_ID:-$(date +%s)}"

mkdir -p "${PF_OUTPUT_DIR}"

args=(
  --train-dataset "${PF_DATA_DIR}/train.jsonl"
  --validation-dataset "${PF_DATA_DIR}/validation.jsonl"
  --test-dataset "${PF_DATA_DIR}/test.jsonl"
  --label-space-json "${PF_DATA_DIR}/label_space.json"
)

if [[ -n "${PF_EXTRA_TRAIN_DATASET:-}" ]]; then
  args+=(--train-dataset "${PF_EXTRA_TRAIN_DATASET}")
fi

args+=(
  --checkpoint "${PF_MODEL}"
  --output-dir "${PF_OUTPUT_DIR}"
  --max-length "${PF_MAX_LENGTH}"
  --epochs "${PF_EPOCHS}"
  --per-device-train-batch-size "${PF_BATCH_SIZE}"
  --per-device-eval-batch-size "${PF_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${PF_GRAD_ACCUM}"
  --learning-rate "${PF_LEARNING_RATE}"
  --lr-scheduler-type "${PF_LR_SCHEDULER_TYPE}"
  --weight-decay "${PF_WEIGHT_DECAY}"
  --warmup-ratio "${PF_WARMUP_RATIO}"
  --max-grad-norm "${PF_MAX_GRAD_NORM}"
  --logging-steps "${PF_LOGGING_STEPS}"
  --save-total-limit "${PF_SAVE_TOTAL_LIMIT}"
  --dataloader-num-workers "${PF_DATALOADER_WORKERS}"
  --seed "${PF_SEED}"
  --early-stopping-patience "${PF_EARLY_STOPPING_PATIENCE}"
)

if [[ -n "${PF_RESUME_FROM_CHECKPOINT:-}" ]]; then
  args+=(--resume-from-checkpoint "${PF_RESUME_FROM_CHECKPOINT}")
else
  args+=(--overwrite-output)
fi
if [[ "${PF_USE_LORA:-0}" == "1" ]]; then
  args+=(--use-lora)
  args+=(--lora-r "${PF_LORA_R:-16}")
  args+=(--lora-alpha "${PF_LORA_ALPHA:-32}")
  args+=(--lora-dropout "${PF_LORA_DROPOUT:-0.05}")
  args+=(--lora-target-modules "${PF_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}")
fi
if [[ -n "${PF_MAX_TRAIN_EXAMPLES:-}" ]]; then
  args+=(--max-train-examples "${PF_MAX_TRAIN_EXAMPLES}")
fi
if [[ -n "${PF_MAX_VALIDATION_EXAMPLES:-}" ]]; then
  args+=(--max-validation-examples "${PF_MAX_VALIDATION_EXAMPLES}")
fi
if [[ -n "${PF_MAX_TEST_EXAMPLES:-}" ]]; then
  args+=(--max-test-examples "${PF_MAX_TEST_EXAMPLES}")
fi

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${PF_NPROC_PER_NODE}" \
  /workspace/train_korean_hf_ddp.py \
  "${args[@]}"
