#!/usr/bin/env bash
# GPU smoke test on the 10 failure cases: convert, split, train (preflight + 2 steps), score, baselines, compare.
# Labels are the cases' dataset labels with a placeholder risk (no API calls); it checks the pipeline, not quality.
# The fixtures include a 10k-token string, so the preflight exercises the worst-case batch memory.
set -euo pipefail
F=tests/fixtures/failure_cases.jsonl
OUT=tests/out_gpu
rm -rf "$OUT" && mkdir -p "$OUT"
step() { echo; echo "=== $* ==="; }

step "1. to_gliner (verified rows built from the fixtures' dataset labels)"
python - <<'PY'
import json
with open("tests/out_gpu/verified.jsonl", "w") as fh:
    for l in open("tests/fixtures/failure_cases.jsonl"):
        r = json.loads(l)
        fh.write(json.dumps({**r, "verified_by": "judge1", "glm_risk": "other" if r["gold_block"] else "no_risk"}) + "\n")
PY
python to_gliner.py --input $OUT/verified.jsonl --output $OUT/gliner.jsonl

step "2. split_data (val only; the benchmark is the test set)"
python split_data.py --input $OUT/gliner.jsonl --manifest $OUT/verified.jsonl --out-prefix $OUT/split --test 0 --val 0.3

step "3. train_gliner: preflight on the longest batch, then 2 optimizer steps"
python train_gliner.py --train $OUT/split_train.jsonl --val $OUT/split_val.jsonl --out $OUT/run \
    --base /models/GLiNER2.5-Decide --max-steps 2 --batch-size 2 --grad-accum 1 --max-len 1024 \
    --gradient-checkpointing --workers 0

step "4. gliner_caller on the 10 cases"
python gliner_caller.py --model $OUT/run --input $F --output $OUT/scored.jsonl

step "5. decision_caller: zero-shot d1-3B, d3-mini and Shieldstral-3B on the 10 cases"
for m in d1 d3 shieldstral; do python decision_caller.py --model $m --input $F --output $OUT/$m.jsonl; done

step "6. train_d3: parity with d3 inference, preflight, 2 LoRA steps, export; then score the package"
python train_d3.py --train $OUT/split_train.jsonl --val $OUT/split_val.jsonl --out $OUT/d3ft --max-steps 2 \
    --batch-size 2 --eval-every 1
python decision_caller.py --model d3 --path $OUT/d3ft/package --input $F --output $OUT/d3ft.jsonl

step "7. compare_systems with the model and both baselines as --system"
python compare_systems.py --bench $F --system smoke=$OUT/scored.jsonl --system d1=$OUT/d1.jsonl \
    --system shieldstral=$OUT/shieldstral.jsonl \
    --system d3ft=$OUT/d3ft.jsonl --cases

python - <<'PY'
import json
for f in ("scored", "d1", "shieldstral", "d3ft"):
    rows = [json.loads(l) for l in open(f"tests/out_gpu/{f}.jsonl")]
    assert len(rows) == 10 and all(0 <= r["block_prob"] <= 1 for r in rows), (f, rows)
summary = json.load(open("tests/out_gpu/run/train-summary.json"))
assert summary["failed_micro_batches"] == 0 and summary["max_len"] == 1024, summary
print("\nSMOKE-GPU OK: converted, split, trained, scored, ran both baselines on and compared all 10 prompts")
PY
rm -rf "$OUT/run/checkpoints" "$OUT/d3ft/package"  # ~1.9 GB and ~9 GB
