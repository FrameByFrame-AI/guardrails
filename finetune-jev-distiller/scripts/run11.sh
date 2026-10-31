#!/bin/bash
# Run 11: d3-lite on run 9's data + conversations, responses, adversarial prompts (build/build_v11.py), A5000; then the
# internal suite (fast GuardBench). Waits for the baseline scorecard job to leave the GPU. Resumable per step.
cd "$(dirname "$0")/.."
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
while pgrep -f "scripts/baseline_card.sh" > /dev/null; do sleep 60; done
echo "=== train $(date -u +%H:%M)"
[ -f runs/d3-lite-v11/package/config.json ] || make train-d3 RUN_NAME=d3-lite-v11 D3_TRAIN=data/v11_train.jsonl \
  D3_VAL=data/v11_val.jsonl D3_ARGS="--base vllm-sr/d3-lite --select loss" > runs/logs/d3-lite-v11-train.log 2>&1
echo "=== suite $(date -u +%H:%M)"
scripts/evalsuite.sh d3-lite-v11 d3 runs/d3-lite-v11/package --fast
echo "=== done $(date -u +%H:%M)"
