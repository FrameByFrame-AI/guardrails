#!/bin/bash
# Every local model on injection benchmark v2 (build/build_injection_v2.py), RTX A5000, resumable; then the table.
# Waits for run 10 to leave the GPU first. Run from finetune-jev-distiller/: scripts/injection_v2.sh
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
while pgrep -f "scripts/run10.sh" > /dev/null; do sleep 60; done
for spec in d3-soup59:d3:runs/d3-soup59/package d3-lite-v9:d3:runs/d3-lite-v9/package d3-lite:d3:vllm-sr/d3-lite \
            gliner-decide:gliner:/models/GLiNER2.5-Decide d3-ft:d3:runs/d3-ft/package d3:d3: xguard-0.6b:xguard-0.6b: \
            xguard:xguard: sing-guard-2b:sing-guard-2b: shieldstral:shieldstral: qwen3guard-0.6b:qwen3guard-0.6b: \
            qwen3guard-4b:qwen3guard-4b: qwen3guard-8b:qwen3guard-8b: haloguard-0.8b:haloguard-0.8b: haloguard-4b:haloguard-4b:; do
  IFS=: read dir model path <<< "$spec"
  [ -n "$path" ] && [ "${path#runs/}" != "$path" ] && [ ! -d "$path" ] && { echo "=== $dir: no package, skipped"; continue; }
  mkdir -p runs/$dir
  echo "=== $dir $(date -u +%H:%M)"
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model $model ${path:+--path $path} \
    --input data/injection_v2.jsonl --output runs/$dir/injection_v2.jsonl > runs/logs/injv2-$dir.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/injection_v2.jsonl)"
done
docker compose run --rm -T -e PYTHONPATH=/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu eval/injection_v2_results.py > runs/injection_v2_results.md
