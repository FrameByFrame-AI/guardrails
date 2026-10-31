#!/bin/bash
# Internal evaluation suite: score one system on every benchmark it has no predictions for yet (resumable).
#   scripts/evalsuite.sh <run dir> <eval/guardrate.py --model> [package path] [--fast]
# GPU: set GPU (default the A5000). --fast scores data/guardbench_fast.jsonl instead of all of GuardBench.
# Then: docker compose run --rm -T -e PYTHONPATH=/workspace/runs/gblibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu eval/scorecard.py --system name=dir
dir=$1; model=$2; path=$3; fast=$4
[ "$path" = "--fast" ] && { path=""; fast="--fast"; }
export GPU=${GPU:-GPU-9ee06423-0765-50b7-644e-2ffc8e65029d}
gb=guardbench; [ "$fast" = "--fast" ] && gb=guardbench_fast
mkdir -p runs/$dir runs/logs
for b in guardrate_en guardrate_ru injection_v2 conv_policy_test:conv_policy $gb; do
  IFS=: read input out <<< "$b"; out=${out:-$input}
  n_in=$(wc -l < data/$input.jsonl); n_out=$( [ -f runs/$dir/$out.jsonl ] && wc -l < runs/$dir/$out.jsonl || echo 0)
  [ "$n_out" -ge "$n_in" ] && { echo "=== $dir $out: done"; continue; }
  echo "=== $dir $out $(date -u +%H:%M)"
  docker compose run --rm -T -e PYTHONPATH=/workspace/runs/pylibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve gpu eval/guardrate.py run --model $model \
    ${path:+--path $path} --input data/$input.jsonl --output runs/$dir/$out.jsonl > runs/logs/suite-$dir-$out.log 2>&1
  echo "exit $? $(date -u +%H:%M) rows $(wc -l < runs/$dir/$out.jsonl)"
done
