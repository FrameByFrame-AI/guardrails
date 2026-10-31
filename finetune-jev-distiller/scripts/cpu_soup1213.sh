#!/bin/bash
# The d3-lite 12+13 soup on CPU only (fp32, as deployed): AA rebuild and injection v2 side by side (6 threads each),
# compared with the GPU (bf16) predictions; then single-request latency at 8 threads. Every step resumes.
cd "$(dirname "$0")/.."
D=runs/d3-lite-soup1213-cpu
mkdir -p $D
echo "=== cpu predictions $(date -u +%H:%M)"
for n in aa_guardrail injection_v2; do
  docker compose run --rm -T -e OMP_NUM_THREADS=6 -e PYTHONPATH=/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu eval/guardrate.py run \
    --model d3 --path runs/d3-lite-soup1213/package --input data/$n.jsonl --output $D/$n.jsonl > runs/logs/cpu-soup1213-$n.log 2>&1 &
done
wait
echo "=== latency $(date -u +%H:%M)"
docker compose run --rm -T -e PYTHONPATH=/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu serve/cpu_latency.py --threads 8 --fp32 --block-only \
  --only "soup" --output runs/cpu_latency.json > runs/logs/cpu-soup1213-latency.log 2>&1
echo "=== done $(date -u +%H:%M)"
