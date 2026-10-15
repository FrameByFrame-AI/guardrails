#!/bin/bash
# Score the run 5 + run 9 weight average: injection benchmark, then GuardRate. Stops at the first failure; both resume.
# Usage, from finetune-jev-distiller/ after make soup: GPU=<uuid> scripts/soup.sh
set -e
: "${GPU:?set GPU to the GPU uuid}"
export GPU
echo "=== injection $(date -u +%H:%M)"
make injection RUN_NAME=d3-soup59 > runs/logs/d3-soup59-injection.log 2>&1
echo "=== guardrate $(date -u +%H:%M)"
make guardrate RUN_NAME=d3-soup59 SEEN=data/guardrate_en_seen_r5r6.json > runs/logs/d3-soup59-guardrate.log 2>&1
echo "=== done $(date -u +%H:%M)"
