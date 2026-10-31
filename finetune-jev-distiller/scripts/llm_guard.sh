#!/bin/bash
# LLM Guard (eval/guardrate.py --model llm-guard) on the internal evals, on CPU with 8 threads (its usual deployment),
# resumable. Run from finetune-jev-distiller/: scripts/llm_guard.sh
mkdir -p runs/llm-guard
for b in injection_v2 guardrate_en guardbench injection_bench; do
  echo "=== $b $(date -u +%H:%M)"
  docker compose run --rm -T -e OMP_NUM_THREADS=8 cpu eval/guardrate.py run --model llm-guard \
    --input data/$b.jsonl --output runs/llm-guard/$b.jsonl > runs/logs/llm-guard-$b.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/llm-guard/$b.jsonl)"
done
