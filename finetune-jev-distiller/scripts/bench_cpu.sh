#!/usr/bin/env bash
# CPU serving benchmark for the Lite 1.2 GGUF: workers x threads layouts on at most 8 physical cores (a 16-vCPU
# cloud VM), a concurrency sweep each, through the same nginx + API key path remote clients use. Resumable.
#   scripts/bench_cpu.sh [bind address]       -> runs/bench_cpu/<layout>.json, summary in runs/bench_cpu/summary.md
# Leaves the default server (1 x 8) running afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."
BIND=${1:-127.0.0.1}
KEY=$(cat runs/serve_cpu/api_key)
OUT=runs/bench_cpu; mkdir -p "$OUT"
LAYOUTS=${LAYOUTS:-"1x8 2x4 4x2 1x4 1x2"}
for l in $LAYOUTS; do
  w=${l%x*}; t=${l#*x}
  [ -f "$OUT/$l.json" ] && { echo "skip $l"; continue; }
  scripts/serve_cpu.sh start --workers "$w" --threads "$t" --bind "$BIND" > "$OUT/$l.serve.log" 2>&1 \
    || { echo "$l: failed to start"; continue; }
  port=$(grep -oE 'port=[0-9]+' "$OUT/$l.serve.log" | cut -d= -f2)
  sweep=1,2,4; [ "$w" -ge 2 ] && sweep=1,$w,$((2 * w))
  echo "== $l (workers $w x threads $t), sweep $sweep"
  python3 serve/latency_cpu.py --url "http://$BIND:$port" --api-key "$KEY" --warmup 10 --duration 30 \
    --sweep "$sweep" --json "$OUT/$l.tmp.json" | tee "$OUT/$l.txt"
  [ -f "$OUT/$l.tmp.json" ] && mv "$OUT/$l.tmp.json" "$OUT/$l.json"
done
scripts/serve_cpu.sh start --bind "$BIND" > /dev/null 2>&1
python3 - "$OUT" <<'EOF'
import json, glob, os, sys
out = sys.argv[1]; rows = []
for f in sorted(glob.glob(f"{out}/*.json")):
    d = json.load(open(f)); tag = os.path.basename(f)[:-5]
    lv = d["levels"]; best = max(lv, key=lambda x: x["throughput_rps"]); c1 = lv[0]
    rows.append(f"| {tag} | {c1['p50_ms']:.0f} | {c1['p95_ms']:.0f} | {c1['throughput_rps']:.2f} | {best['throughput_rps']:.2f} | "
                f"{best['concurrency']} | {best['p50_ms']:.0f} |")
hdr = ("| workers x threads | c=1 p50 ms | c=1 p95 ms | c=1 req/s | best req/s | at concurrency | p50 ms there |\n"
       "|---|---|---|---|---|---|---|")
open(f"{out}/summary.md", "w").write(hdr + "\n" + "\n".join(rows) + "\n"); print(hdr); print("\n".join(rows))
EOF
