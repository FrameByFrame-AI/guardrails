#!/bin/bash
# Artificial Analysis guardrail benchmark, local rebuild (build/build_aa.py), A5000, resumable; then the table.
# Waits for run 11 to leave the GPU. Run from finetune-jev-distiller/: scripts/aa.sh
cd "$(dirname "$0")/.."
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
while pgrep -f "^/bin/bash scripts/run11.sh" > /dev/null; do sleep 60; done
for spec in d3-soup59:d3:runs/d3-soup59/package d3-lite-v11:d3:runs/d3-lite-v11/package d3-lite-v9:d3:runs/d3-lite-v9/package \
            d3:d3: qwen3guard-8b:qwen3guard-8b: shieldstral:shieldstral: xguard:xguard: llm-guard:llm-guard:; do
  IFS=: read dir model path <<< "$spec"
  n=$( [ -f runs/$dir/aa_guardrail.jsonl ] && wc -l < runs/$dir/aa_guardrail.jsonl || echo 0)
  [ "$n" -ge 7232 ] && continue
  echo "=== $dir $(date -u +%H:%M)"
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model $model \
    ${path:+--path $path} --input data/aa_guardrail.jsonl --output runs/$dir/aa_guardrail.jsonl > runs/logs/aa-$dir.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/aa_guardrail.jsonl)"
done
S=(); for x in "AccuKnox Guard 1.1:d3-soup59" "run 11 (d3-lite):d3-lite-v11" "run 10 (d3-lite):d3-lite-v9" "d3-mini untrained:d3" \
  "Qwen3Guard-8B:qwen3guard-8b" "Shieldstral-3B:shieldstral" "XGuard-8B:xguard" "stock LLM Guard:llm-guard"; do
  S+=(--system "${x%%:*}=runs/${x##*:}/aa_guardrail.jsonl"); done
{ docker compose run --rm -T cpu build/build_aa.py score "${S[@]}"; echo; docker compose run --rm -T cpu build/build_aa.py score "${S[@]}" --seen data/aa_guardrail_seen.json; } > runs/aa_results.md 2>&1
