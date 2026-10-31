#!/bin/bash
# d3-mini run 12 (1.2 parent): internal suite (fast GuardBench) and the AA rebuild on the A5000, while run 13 trains on
# the RTX PRO 6000. Resumable.
cd "$(dirname "$0")/.."
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
scripts/evalsuite.sh d3-mini-v12 d3 runs/d3-mini-v12/package --fast
[ -f runs/d3-mini-v12/aa_guardrail.jsonl ] && [ $(wc -l < runs/d3-mini-v12/aa_guardrail.jsonl) -eq 7232 ] || \
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model d3 \
  --path runs/d3-mini-v12/package --input data/aa_guardrail.jsonl --output runs/d3-mini-v12/aa_guardrail.jsonl \
  > runs/logs/aa-d3-mini-v12.log 2>&1
echo "=== done $(date -u +%H:%M)"
