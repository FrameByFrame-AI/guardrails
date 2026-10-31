#!/bin/bash
# CPU-model candidates, untrained, on GuardRate English and the injection benchmark, RTX A5000, resumable.
# d3-lite comes from the Hub at its current revision; GLiNER2.5-Decide from /models.
export GPU=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d
for spec in d3-lite:d3:vllm-sr/d3-lite gliner-decide:gliner:/models/GLiNER2.5-Decide; do
  IFS=: read dir model path <<< "$spec"
  mkdir -p runs/$dir
  for bench in guardrate_en injection_bench; do
    echo "=== $dir $bench $(date -u +%H:%M)"
    docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model $model \
      --path $path --input data/$bench.jsonl --output runs/$dir/$bench.jsonl > runs/logs/cpu-cand-$dir-$bench.log 2>&1
    echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/$bench.jsonl)"
  done
done
