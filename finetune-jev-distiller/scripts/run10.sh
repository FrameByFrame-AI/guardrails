#!/bin/bash
# Run 10 (CPU model) on the RTX A5000: d3-lite fine-tuned like run 9 (resumable), then the injection benchmark and
# GuardRate. Stops at the first failure. Run from finetune-jev-distiller/: scripts/run10.sh
set -e
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
echo "=== train $(date -u +%H:%M)"
make train-d3 RUN_NAME=d3-lite-v9 D3_TRAIN=data/v9_train.jsonl D3_VAL=data/v9_val.jsonl \
  D3_ARGS="--base vllm-sr/d3-lite --select loss" > runs/logs/d3-lite-v9-train.log 2>&1
echo "=== injection $(date -u +%H:%M)"
make injection RUN_NAME=d3-lite-v9 > runs/logs/d3-lite-v9-injection.log 2>&1
echo "=== guardrate $(date -u +%H:%M)"
make guardrate RUN_NAME=d3-lite-v9 SEEN=data/guardrate_en_seen_r5r6.json > runs/logs/d3-lite-v9-guardrate.log 2>&1
echo "=== done $(date -u +%H:%M)"
