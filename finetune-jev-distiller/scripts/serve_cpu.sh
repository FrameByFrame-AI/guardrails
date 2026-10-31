#!/usr/bin/env bash
# Serve AccuKnox Guard Lite 1.2 (Q8_0 GGUF) on this machine's CPU for remote latency tests (serve/latency_bench.py).
#
#   scripts/serve_cpu.sh start                       # 1 worker x 8 threads, 127.0.0.1:47613, API key required
#   scripts/serve_cpu.sh start --bind 0.0.0.0        # accept remote clients (or a LAN / VPN address)
#   scripts/serve_cpu.sh start --workers 2 --threads 4
#   scripts/serve_cpu.sh status | stop
#
# Layout: N llama-server workers (llama.cpp official image, pinned by digest), each pinned to its own physical
# cores and memory-capped, behind one nginx on --port that sends each request to the least busy worker. Every
# response carries X-Guard-Workers / X-Guard-Threads / X-Guard-Slots so the client can print the server setup.
#
# Defaults are the measured best for one-request latency: 1 worker, 8 threads (= free physical cores), 1 slot.
# For throughput, more workers on disjoint cores beat more threads or slots (docs/experiments.md, 2026-10-19):
# 2 x 4 threads gives ~17% more requests/s, each request ~1.7x slower; slots (-np) add nothing on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=1          # llama-server processes
THREADS=8          # threads per worker; workers x threads must not exceed the free physical cores
SLOTS=1            # parallel slots per worker (-np); 1 is best on CPU
MEM_GB=4           # memory cap per worker (peak measured 2.4 GiB at a 5,400-token input)
CTX=8192           # context = batch = ubatch: the longest input in tokens (d3 reads the prompt in one batch)
PORT=47613         # uncommon high port, away from usual dev ports
BIND=127.0.0.1     # local only; --bind 0.0.0.0 (or one interface's address) for remote clients
API_KEY=""         # default: generated once and kept in runs/serve_cpu/api_key
FIRST_CORE=""      # default: highest cores first, leaving low cores to the OS and other jobs
MODEL=hf/accuknox-guard-lite-1.2-GGUF/accuknox-guard-lite-1.2-Q8_0.gguf
IMAGE=ghcr.io/ggml-org/llama.cpp@sha256:c7b03e45cff9c4e984acd39032a6f3a82c811075b72bf176e329f84b8598c135
NGINX=nginx:1.27.3-alpine
NET=guard-cpu
DIR=runs/serve_cpu

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; cat <<EOF

Options (start): --workers N ($WORKERS) --threads T ($THREADS) --slots S ($SLOTS) --memory-gb G ($MEM_GB)
                 --ctx TOKENS ($CTX) --port P ($PORT) --bind ADDR ($BIND) --api-key KEY (generated)
                 --first-core C (auto) --no-key (no authentication)
EOF
}

stop() {
  { docker ps -a --format '{{.Names}}' | grep -E '^guard-cpu-(w[0-9]+|lb)$' || true; } | xargs -r docker rm -f >/dev/null
  docker network rm "$NET" >/dev/null 2>&1 || true
  echo "stopped"
}

status() {
  docker ps --filter name=guard-cpu- --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  [ -f "$DIR/config" ] && cat "$DIR/config"
}

cmd=${1:-help}; shift || true
NO_KEY=0
while [ $# -gt 0 ]; do
  case $1 in
    --workers) WORKERS=$2; shift 2;; --threads) THREADS=$2; shift 2;; --slots) SLOTS=$2; shift 2;;
    --memory-gb) MEM_GB=$2; shift 2;; --ctx) CTX=$2; shift 2;; --port) PORT=$2; shift 2;;
    --bind) BIND=$2; shift 2;; --api-key) API_KEY=$2; shift 2;; --first-core) FIRST_CORE=$2; shift 2;;
    --no-key) NO_KEY=1; shift;;
    *) echo "unknown option $1"; usage; exit 2;;
  esac
done

case $cmd in
  stop) stop; exit 0;;
  status) status; exit 0;;
  start) ;;
  *) usage; exit 0;;
esac

[ -f "$MODEL" ] || { echo "missing $MODEL"; exit 1; }
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -un | wc -l)
NEED=$((WORKERS * THREADS))
[ "$NEED" -le "$CORES" ] || { echo "workers x threads = $NEED exceeds the $CORES physical cores"; exit 1; }
[ -n "$FIRST_CORE" ] || FIRST_CORE=$((CORES - NEED))
mkdir -p "$DIR"
if [ "$NO_KEY" = 0 ] && [ -z "$API_KEY" ]; then
  [ -f "$DIR/api_key" ] || python3 -c "import secrets; print(secrets.token_urlsafe(24))" > "$DIR/api_key"
  API_KEY=$(cat "$DIR/api_key")
fi

stop >/dev/null
docker network create "$NET" >/dev/null

# physical core c is logical CPU c (and its SMT sibling c + CORES); pin one thread per physical core
UPSTREAM=""
for ((w = 0; w < WORKERS; w++)); do
  start=$((FIRST_CORE + w * THREADS)); end=$((start + THREADS - 1))
  docker run -d --name "guard-cpu-w$w" --network "$NET" --restart unless-stopped \
    --cpuset-cpus "$start-$end" --memory "${MEM_GB}g" --memory-swap "${MEM_GB}g" \
    -v "$PWD/$(dirname "$MODEL"):/models:ro" "$IMAGE" \
    -m "/models/$(basename "$MODEL")" --host 0.0.0.0 --port 8080 \
    -t "$THREADS" -tb "$THREADS" -c "$((CTX * SLOTS))" -b "$CTX" -ub "$CTX" -np "$SLOTS" \
    ${API_KEY:+--api-key "$API_KEY"} >/dev/null
  UPSTREAM+="    server guard-cpu-w$w:8080 max_fails=0;"$'\n'
  echo "worker $w: cores $start-$end, $THREADS threads, $SLOTS slot(s), ${MEM_GB} GB"
done

cat > "$DIR/nginx.conf" <<EOF
events { worker_connections 1024; }
http {
  upstream guard { least_conn;
$UPSTREAM  }
  server {
    listen 8100;
    client_max_body_size 16m;
    location / {
      proxy_pass http://guard;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_read_timeout 600s;
      add_header X-Guard-Workers $WORKERS always;
      add_header X-Guard-Threads $THREADS always;
      add_header X-Guard-Slots $SLOTS always;
      add_header X-Guard-Memory-GB $MEM_GB always;
    }
  }
}
EOF
docker run -d --name guard-cpu-lb --network "$NET" --restart unless-stopped -p "$BIND:$PORT:8100" \
  -v "$PWD/$DIR/nginx.conf:/etc/nginx/nginx.conf:ro" "$NGINX" >/dev/null

printf "workers=%s threads=%s slots=%s memory_gb=%s ctx=%s cores=%s-%s port=%s bind=%s\n" \
  "$WORKERS" "$THREADS" "$SLOTS" "$MEM_GB" "$CTX" "$FIRST_CORE" "$((FIRST_CORE + NEED - 1))" "$PORT" "$BIND" > "$DIR/config"
echo -n "loading"
for i in $(seq 120); do
  ready=0
  for ((w = 0; w < WORKERS; w++)); do
    docker exec "guard-cpu-w$w" curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && ready=$((ready + 1))
  done
  [ "$ready" = "$WORKERS" ] && break; echo -n "."; sleep 1
done
echo; [ "$ready" = "$WORKERS" ] || { echo "workers not ready after 120 s: docker logs guard-cpu-w0"; exit 1; }
cat "$DIR/config"
HOSTS=$(hostname -I | tr ' ' '\n' | grep -E '^(192\.168|10\.|100\.)' | head -2 | tr '\n' ' ')
echo "ready: http://$BIND:$PORT$( [ "$BIND" = 0.0.0.0 ] && echo "  (this machine: $HOSTS)")"
[ -n "$API_KEY" ] && echo "api key: $DIR/api_key  (pass it as --api-key or GUARD_API_KEY)"
