#!/usr/bin/env bash
# Serve AccuKnox Guard 1.2 or Lite 1.2 on one GPU with vLLM SR's runtime (vllm-srun), for local or remote latency
# tests (serve/latency_cpu.py works against any System One server).
#
#   scripts/serve_gpu.sh start --model lite --gpu a5000                 # 127.0.0.1:<port>, API key required
#   scripts/serve_gpu.sh start --model 1.2 --gpu blackwell --replicas 2 --bind 0.0.0.0
#   scripts/serve_gpu.sh status | stop [--model M --gpu G]
#
# Layout: R vllm-srun replicas on the same GPU behind one nginx on --port, least-busy routing; nginx checks the
# API key (vllm-srun has none) and adds X-Guard-* headers with the setup. bf16, offline, manifest-verified.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=lite         # lite | 1.2
GPU=a5000          # a5000 | blackwell | a GPU UUID
REPLICAS=1         # vllm-srun processes on the GPU
PORT=""            # default: 47621 (1.2) / 47622 (lite) on the A5000, 47631 / 47632 on the Blackwell
BIND=127.0.0.1     # local only; --bind 0.0.0.0 (or one interface's address) for remote clients
API_KEY=""         # default: the CPU server's key (runs/serve_cpu/api_key), else generated
NO_KEY=0
IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:e38f1fe79be9731d192622a6a982791e405b3666
NGINX=nginx:1.27.3-alpine

cmd=${1:-help}; shift || true
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2;; --gpu) GPU=$2; shift 2;; --replicas) REPLICAS=$2; shift 2;;
    --port) PORT=$2; shift 2;; --bind) BIND=$2; shift 2;; --api-key) API_KEY=$2; shift 2;; --no-key) NO_KEY=1; shift;;
    *) echo "unknown option $1"; sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2;;
  esac
done

case $GPU in
  a5000) UUID=GPU-9ee06423-0765-50b7-644e-2ffc8e65029d; G=a5000; BASEPORT=47620;;
  blackwell) UUID=GPU-8aad1aba-0ade-610e-fa0d-6744ac28d274; G=blackwell; BASEPORT=47630;;
  GPU-*) UUID=$GPU; G=${GPU:4:8}; BASEPORT=47640;;
  *) echo "--gpu a5000 | blackwell | GPU-<uuid>"; exit 2;;
esac
case $MODEL in
  1.2) PKG=hf/accuknox-guard-1.2; NAME=accuknox-guard-1.2; M=12; OFF=1;;
  lite) PKG=hf/accuknox-guard-lite-1.2; NAME=accuknox-guard-lite-1.2; M=lite; OFF=2;;
  *) echo "--model lite | 1.2"; exit 2;;
esac
[ -n "$PORT" ] || PORT=$((BASEPORT + OFF))
P=guard-gpu-$M-$G

stop() {
  { docker ps -a --format '{{.Names}}' | grep -E "^$1-(r[0-9]+|lb)\$" || true; } | xargs -r docker rm -f >/dev/null
  docker network rm "$1" >/dev/null 2>&1 || true
}

case $cmd in
  stop)
    if [ "$*" = "" ] && [ "$MODEL$GPU" = "litea5000" ]; then  # no selector: stop every GPU deployment
      for n in $(docker ps -a --format '{{.Names}}' | grep -oE '^guard-gpu-[a-z0-9]+-[a-z0-9]+' | sort -u); do stop "$n"; done
    else stop "$P"; fi
    echo stopped; exit 0;;
  status) docker ps --filter name=guard-gpu- --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'; cat runs/serve_gpu/*.config 2>/dev/null || true; exit 0;;
  start) ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
esac

[ -f "$PKG/MODEL_MANIFEST.json" ] || { echo "$PKG has no MODEL_MANIFEST.json"; exit 1; }
mkdir -p runs/serve_gpu
if [ "$NO_KEY" = 0 ] && [ -z "$API_KEY" ]; then
  if [ -f runs/serve_cpu/api_key ]; then API_KEY=$(cat runs/serve_cpu/api_key)
  else [ -f runs/serve_gpu/api_key ] || python3 -c "import secrets; print(secrets.token_urlsafe(24))" > runs/serve_gpu/api_key
       API_KEY=$(cat runs/serve_gpu/api_key); fi
fi

stop "$P"
docker network create "$P" >/dev/null
UPSTREAM=""
for ((r = 0; r < REPLICAS; r++)); do
  docker run -d --name "$P-r$r" --network "$P" --restart unless-stopped --gpus "device=$UUID" \
    -v "$PWD/$PKG:/models/$NAME:ro" --entrypoint vllm-srun "$IMAGE" serve "/models/$NAME" \
    --host 0.0.0.0 --port 8100 --device cuda --offline --served-model-name "$NAME" >/dev/null
  UPSTREAM+="    server $P-r$r:8100 max_fails=0;"$'\n'
done

AUTH=""
[ -n "$API_KEY" ] && AUTH="if (\$http_authorization != \"Bearer $API_KEY\") { return 401; }"
cat > "runs/serve_gpu/$P.nginx.conf" <<EOF
events { worker_connections 4096; }
http {
  upstream guard { least_conn;
$UPSTREAM    keepalive 64;
  }
  server {
    listen 8100;
    client_max_body_size 16m;
    location = /health { proxy_pass http://guard; }
    location / {
      $AUTH
      proxy_pass http://guard;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_read_timeout 600s;
      add_header X-Guard-Model $NAME always;
      add_header X-Guard-GPU $G always;
      add_header X-Guard-Workers $REPLICAS always;
    }
  }
}
EOF
chmod 600 "runs/serve_gpu/$P.nginx.conf"
docker run -d --name "$P-lb" --network "$P" --restart unless-stopped -p "$BIND:$PORT:8100" \
  -v "$PWD/runs/serve_gpu/$P.nginx.conf:/etc/nginx/nginx.conf:ro" "$NGINX" >/dev/null

echo -n "loading $NAME x$REPLICAS on $G"
for i in $(seq 120); do
  ready=0
  for ((r = 0; r < REPLICAS; r++)); do
    docker exec "$P-r$r" python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health')" \
      >/dev/null 2>&1 && ready=$((ready + 1))
  done
  [ "$ready" = "$REPLICAS" ] && break
  if docker logs "$P-r0" 2>&1 | grep -qE "Traceback|CUDA out of memory"; then echo; docker logs "$P-r0" 2>&1 | tail -5; exit 1; fi
  echo -n "."; sleep 5
done
echo; [ "$ready" = "$REPLICAS" ] || { echo "not ready after 10 min: docker logs $P-r0"; exit 1; }
printf "%s: model=%s gpu=%s replicas=%s port=%s bind=%s\n" "$P" "$NAME" "$G" "$REPLICAS" "$PORT" "$BIND" | tee "runs/serve_gpu/$P.config"
echo "ready: http://$BIND:$PORT"
