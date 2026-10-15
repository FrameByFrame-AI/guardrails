#!/usr/bin/env bash
# Smoke test on the 10 failure cases in tests/fixtures: every API module end to end, fresh caches in tests/out.
# Costs about $0.001 per run (10 calls each to Jev and GLM 5.3 Flash, 10 Bedrock text units).
set -euo pipefail
F=tests/fixtures/failure_cases.jsonl
OUT=tests/out
rm -rf "$OUT" && mkdir -p "$OUT"
step() { echo; echo "=== $* ==="; }

step "1. datasets_spec: every dataset's splits, columns and label mapping load from the cache"
python - <<'PY'
import random
from datasets_spec import DATASETS, load_rows
for name, spec in DATASETS.items():
    for which in ("bench", "train"):
        if spec[which]:
            rows = load_rows(name, which, 2, random.Random(0))
            # beavertails has no usable prompt label by design, so it yields no rows
            assert rows or name == "beavertails", f"{name}/{which}: no labelled rows"
            print(f"  {name:<20} {which:<5} ok ({len(rows)} labelled rows sampled)")
PY

step "2. build_benchmark: 2 candidates per benchmark dataset"
python build_benchmark.py --per-dataset 2 --output $OUT/candidates.jsonl

step "3. jev_caller"
python jev_caller.py --input $F --output $OUT/jev.jsonl

step "4. verify_glm (GLM 5.3 Flash vs dataset labels; no dataset dropped at this size)"
python verify_glm.py --input $F --cache $OUT/glm.jsonl --output $OUT/verified.jsonl \
    --disputed $OUT/disputed.jsonl --min-dataset-agreement 0

step "5. bedrock_caller"
python bedrock_caller.py --input $F --cache $OUT/bedrock.jsonl

step "6. compare_systems on the 10 cases (dataset labels)"
python compare_systems.py --bench $F --jev $OUT/jev.jsonl --glm $OUT/glm.jsonl --bedrock $OUT/bedrock.jsonl \
    --cases --output $OUT/compare.json

python - <<'PY'
import json
answered = lambda name, keep=lambda r: True: sum(keep(json.loads(l)) for l in open(f"tests/out/{name}.jsonl"))
for name, keep in (("jev", lambda r: True), ("glm", lambda r: r["model"] == "z-ai/glm-5.3-flash"), ("bedrock", lambda r: True)):
    n = answered(name, keep)
    assert n == 10, f"{name}: {n} of 10 prompts answered"
print("\nSMOKE OK: all modules ran; Jev, GLM and Bedrock answered all 10 prompts")
PY
