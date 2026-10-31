#!/usr/bin/env bash
# GPU serving benchmark: each model x GPU x replica count, one after another (no other GPU work), a concurrency
# sweep per setup through the same nginx + API key path remote clients use. Resumable: finished setups are skipped.
#   scripts/bench_gpu.sh [bind address]          -> runs/bench_gpu/<setup>.json, summary in runs/bench_gpu/summary.md
set -uo pipefail
cd "$(dirname "$0")/.."
BIND=${1:-127.0.0.1}
KEY=$(cat runs/serve_cpu/api_key)
OUT=runs/bench_gpu; mkdir -p "$OUT"
SETUPS=${SETUPS:-"lite:a5000:1 lite:a5000:2 1.2:a5000:1 lite:blackwell:1 lite:blackwell:2 lite:blackwell:4 1.2:blackwell:1 1.2:blackwell:2 1.2:blackwell:4"}
for s in $SETUPS; do
  IFS=: read -r model gpu reps <<< "$s"
  tag="$model-$gpu-x$reps"
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; continue; }
  scripts/serve_gpu.sh stop >/dev/null
  if ! scripts/serve_gpu.sh start --model "$model" --gpu "$gpu" --replicas "$reps" --bind "$BIND" > "$OUT/$tag.serve.log" 2>&1; then
    echo "$tag: failed to start (see $OUT/$tag.serve.log)"; continue
  fi
  port=$(grep -oE 'port=[0-9]+' "$OUT/$tag.serve.log" | cut -d= -f2)
  id=$([ "$gpu" = a5000 ] && echo 9ee06423 || echo 8aad1aba)
  mem=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | grep "$id" | cut -d, -f2)
  echo "== $tag on port $port, GPU memory$mem"
  python3 serve/latency_cpu.py --url "http://$BIND:$port" --api-key "$KEY" --warmup 30 --duration 15 \
    --sweep 1,2,4,8,16,32,64 --json "$OUT/$tag.tmp.json" | tee "$OUT/$tag.txt"
  [ -f "$OUT/$tag.tmp.json" ] && mv "$OUT/$tag.tmp.json" "$OUT/$tag.json" && echo "gpu_memory:$mem" >> "$OUT/$tag.txt"
done
scripts/serve_gpu.sh stop >/dev/null
python3 - "$OUT" <<'EOF'
import json, glob, os, sys
out = sys.argv[1]; rows = []
for f in sorted(glob.glob(f"{out}/*.json")):
    d = json.load(open(f)); tag = os.path.basename(f)[:-5]
    lv = d["levels"]; best = max(lv, key=lambda x: x["throughput_rps"])
    c1 = next(x for x in lv if x["concurrency"] == 1)
    rows.append(f"| {tag} | {c1['p50_ms']:.1f} | {c1['throughput_rps']:.0f} | {best['throughput_rps']:.0f} | {best['concurrency']} | {best['p50_ms']:.0f} | {best['p95_ms']:.0f} |")
hdr = "| setup | c=1 p50 ms | c=1 req/s | best req/s | at concurrency | p50 ms there | p95 ms there |\n|---|---|---|---|---|---|---|"
open(f"{out}/summary.md", "w").write(hdr + "\n" + "\n".join(rows) + "\n"); print(hdr); print("\n".join(rows))
EOF
