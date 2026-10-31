#!/bin/bash
# Fill the baseline scorecard on the A5000 (shared with RSNA), resumable; then print it to runs/scorecard.md.
# XGuard-8B (~17 GB) does not fit next to RSNA's 8 GB: its missing columns wait for a free GPU.
cd "$(dirname "$0")/.."
scripts/evalsuite.sh d3-soup59 d3 runs/d3-soup59/package
scripts/evalsuite.sh d3-lite-v9 d3 runs/d3-lite-v9/package
scripts/evalsuite.sh llm-guard llm-guard --fast
scripts/evalsuite.sh d3-lite d3 vllm-sr/d3-lite --fast
docker compose run --rm -T -e PYTHONPATH=/workspace/runs/gblibs:/workspace/lib:/workspace/callers:/workspace/build:/workspace/train:/workspace/eval:/workspace/serve cpu eval/scorecard.py \
  --system "AccuKnox Guard 1.1=d3-soup59" --system "run 10 (d3-lite)=d3-lite-v9" --system "d3-lite untrained=d3-lite" \
  --system "XGuard-8B=xguard" --system "stock LLM Guard=llm-guard" > runs/scorecard.md
