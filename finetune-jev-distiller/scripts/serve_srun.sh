#!/bin/bash
# Serve a d3 package with vLLM Semantic Router's model runtime (vllm-srun, the engine behind `vllm-sr serve -e`)
# in its CUDA image, on one GPU, bound to localhost; System One API at http://127.0.0.1:<port>/v1/systemone.
#   scripts/serve_srun.sh <package dir> [name] [port] [GPU uuid]     (stop: docker rm -f srun_<name>)
# The package needs MODEL_MANIFEST.json (train/d3_manifest.py; train/train_d3.py and train/merge_runs.py write it). Then e.g.
#   python3 serve/srun_bench.py http://127.0.0.1:8100 runs/_loadtest/lat300.jsonl runs/<run>/guardrate_en.jsonl 1 /dev/null
cd "$(dirname "$0")/.."
pkg=$(realpath "$1"); name=${2:-$(basename "$(dirname "$pkg")")}; port=${3:-8100}
gpu=${4:-GPU-9ee06423-0765-50b7-644e-2ffc8e65029d}  # the A5000
IMAGE=${SRUN_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:e38f1fe79be9731d192622a6a982791e405b3666}
[ -f "$pkg/MODEL_MANIFEST.json" ] || { echo "$pkg has no MODEL_MANIFEST.json: python3 train/d3_manifest.py $pkg"; exit 1; }
docker run -d --name "srun_$name" --gpus "device=$gpu" -p "127.0.0.1:$port:8100" -v "$pkg:/models/$name:ro" \
  --entrypoint vllm-srun "$IMAGE" serve "/models/$name" --host 0.0.0.0 --port 8100 --device cuda --offline \
  --served-model-name "$name" > /dev/null || exit 1
for i in $(seq 120); do
  curl -sf "127.0.0.1:$port/health" > /dev/null && { docker logs "srun_$name" 2>&1 | grep "serving"; exit 0; }
  docker logs "srun_$name" 2>&1 | grep -E "ERROR|Traceback" && exit 1
  sleep 5
done
echo "srun_$name not ready after 10 min"; exit 1
