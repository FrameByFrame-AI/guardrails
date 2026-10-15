#!/bin/bash
# p50 latency for every local model on one GPU (RTX A5000), same 1,000-item sample (guardrate.py --limit), resumable.
# Run from finetune-jev-distiller/: scripts/latency_a5000.sh
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
for spec in d3-soup59:d3:runs/d3-soup59/package d3-ft:d3:runs/d3-ft/package d3:d3: xguard:xguard: xguard-0.6b:xguard-0.6b: qwen3guard-8b:qwen3guard-8b: \
            qwen3guard-4b:qwen3guard-4b: qwen3guard-0.6b:qwen3guard-0.6b: haloguard-4b:haloguard-4b: \
            haloguard-0.8b:haloguard-0.8b: sing-guard-2b:sing-guard-2b: shieldstral:shieldstral:; do
  IFS=: read dir model path <<< "$spec"
  echo "=== $dir $(date -u +%H:%M)"
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs gpu guardrate.py run --model $model ${path:+--path $path} \
    --limit 1000 --input data/guardrate_en.jsonl --output runs/$dir/latency_a5000.jsonl > runs/logs/latency-$dir.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/latency_a5000.jsonl)"
done
