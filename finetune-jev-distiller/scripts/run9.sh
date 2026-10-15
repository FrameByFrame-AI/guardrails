#!/bin/bash
# Run 9 on the RTX PRO 6000: train (resumable), then the injection benchmark and GuardRate. Stops at the first failure.
# Run from finetune-jev-distiller/: scripts/run9.sh
set -e
export GPU=GPU-8aad1aba-0ade-610e-fa0d-6744ac28d274
echo "=== train $(date -u +%H:%M)"
make train-d3 RUN_NAME=d3-v9 D3_TRAIN=data/v9_train.jsonl D3_VAL=data/v9_val.jsonl D3_ARGS="--select loss" > runs/logs/d3-v9-train.log 2>&1
echo "=== injection $(date -u +%H:%M)"
make injection RUN_NAME=d3-v9 > runs/logs/d3-v9-injection.log 2>&1
echo "=== guardrate $(date -u +%H:%M)"
make guardrate RUN_NAME=d3-v9 SEEN=data/guardrate_en_seen_r5r6.json > runs/logs/d3-v9-guardrate.log 2>&1
echo "=== done $(date -u +%H:%M)"
