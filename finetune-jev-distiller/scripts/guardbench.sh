#!/bin/bash
# GuardBench (build/build_guardbench.py) on AccuKnox Guard 1.1 and run 10, RTX A5000, resumable; then the table.
# Run from finetune-jev-distiller/: scripts/guardbench.sh
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
for dir in d3-soup59 d3-lite-v9; do
  echo "=== $dir $(date -u +%H:%M)"
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model d3 \
    --path runs/$dir/package --input data/guardbench.jsonl --output runs/$dir/guardbench.jsonl > runs/logs/gb-$dir.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/guardbench.jsonl)"
done
docker compose run --rm -T -e PYTHONPATH=/workspace/runs/gblibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu build/build_guardbench.py score \
  --input data/guardbench.jsonl --system "AccuKnox Guard 1.1=runs/d3-soup59/guardbench.jsonl" \
  --system "run 10=runs/d3-lite-v9/guardbench.jsonl" > runs/guardbench_results.md
