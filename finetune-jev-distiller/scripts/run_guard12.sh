#!/bin/bash
# AccuKnox Guard 1.2: d3-mini on data/v12_* then data/v13_* on the RTX PRO 6000 (two runs side by side were slower:
# 2.8 records/s together vs 3.8 alone), then their
# equal-weight soup, the internal suite (fast GuardBench) and the AA rebuild on the soup. Every step resumes.
cd "$(dirname "$0")/.."
export GPU=GPU-8aad1aba-0ade-610e-fa0d-6744ac28d274
echo "=== train $(date -u +%H:%M)"
for v in 12 13; do
  [ -f runs/d3-mini-v$v/package/config.json ] || make train-d3 RUN_NAME=d3-mini-v$v D3_TRAIN=data/v${v}_train.jsonl \
    D3_VAL=data/v${v}_val.jsonl D3_ARGS="--select loss" >> runs/logs/d3-mini-v$v-train.log 2>&1
done
for v in 12 13; do [ -f runs/d3-mini-v$v/package/config.json ] || { echo "d3-mini-v$v did not finish"; exit 1; }; done
echo "=== merge $(date -u +%H:%M)"
[ -f runs/d3-mini-soup1213/package/config.json ] || \
  make soup SOUP_RUNS="d3-mini-v12 d3-mini-v13" SOUP_NAME=d3-mini-soup1213 > runs/logs/d3-mini-soup1213-merge.log 2>&1 || exit 1
echo "=== suite $(date -u +%H:%M)"
scripts/evalsuite.sh d3-mini-soup1213 d3 runs/d3-mini-soup1213/package --fast
[ -f runs/d3-mini-soup1213/aa_guardrail.jsonl ] && [ $(wc -l < runs/d3-mini-soup1213/aa_guardrail.jsonl) -eq 7232 ] || \
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model d3 \
  --path runs/d3-mini-soup1213/package --input data/aa_guardrail.jsonl --output runs/d3-mini-soup1213/aa_guardrail.jsonl \
  > runs/logs/aa-d3-mini-soup1213.log 2>&1
echo "=== done $(date -u +%H:%M)"
