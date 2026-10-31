#!/bin/bash
# d3-lite soup: equal-weight average of runs 12 and 13 (train/merge_runs.py, CPU), then the internal suite (fast GuardBench)
# and the AA rebuild on the A5000. Every step resumes.
cd "$(dirname "$0")/.."
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
echo "=== merge $(date -u +%H:%M)"
make soup SOUP_RUNS="d3-lite-v12 d3-lite-v13" SOUP_NAME=d3-lite-soup1213 > runs/logs/d3-lite-soup1213-merge.log 2>&1 || exit 1
echo "=== suite $(date -u +%H:%M)"
scripts/evalsuite.sh d3-lite-soup1213 d3 runs/d3-lite-soup1213/package --fast
[ -f runs/d3-lite-soup1213/aa_guardrail.jsonl ] && [ $(wc -l < runs/d3-lite-soup1213/aa_guardrail.jsonl) -eq 7232 ] || \
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model d3 \
  --path runs/d3-lite-soup1213/package --input data/aa_guardrail.jsonl --output runs/d3-lite-soup1213/aa_guardrail.jsonl \
  > runs/logs/aa-d3-lite-soup1213.log 2>&1
echo "=== done $(date -u +%H:%M)"
