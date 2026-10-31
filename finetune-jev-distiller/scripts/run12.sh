#!/bin/bash
# Run 12: d3-lite on data/v12_* (build/build_v12.py), A5000; then the internal suite (fast GuardBench) and the AA rebuild.
cd "$(dirname "$0")/.."
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
echo "=== train $(date -u +%H:%M)"
[ -f runs/d3-lite-v12/package/config.json ] || make train-d3 RUN_NAME=d3-lite-v12 D3_TRAIN=data/v12_train.jsonl \
  D3_VAL=data/v12_val.jsonl D3_ARGS="--base vllm-sr/d3-lite --select loss" > runs/logs/d3-lite-v12-train.log 2>&1
echo "=== suite $(date -u +%H:%M)"
scripts/evalsuite.sh d3-lite-v12 d3 runs/d3-lite-v12/package --fast
docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model d3 \
  --path runs/d3-lite-v12/package --input data/aa_guardrail.jsonl --output runs/d3-lite-v12/aa_guardrail.jsonl > runs/logs/aa-d3-lite-v12.log 2>&1
echo "=== done $(date -u +%H:%M)"
