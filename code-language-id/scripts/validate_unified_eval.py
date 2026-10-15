#!/usr/bin/env python3
"""Independent validation of the unified 113-language eval parquet.

Re-derives coverage and leak status from the written file (not the builder's
in-memory state):
  - every one of the 113 canonical languages is present, with per-kind counts
  - zero prompt overlap with the genuine training sources
  - kind distribution summary
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import LANGUAGES, resolve  # noqa: E402
from build_unified_eval import (  # noqa: E402
    ACCUKNOX, V1_SPLITS, ACCUKNOX_TRAIN_GLOBS, norm_hash,
)

EVAL = ACCUKNOX / "eval-unified-113-00000-of-00001.parquet"


def main() -> int:
    df = pd.read_parquet(EVAL)
    print(f"eval rows = {len(df):,}  kinds = {dict(Counter(df['kind']))}")

    # per-language per-kind coverage
    per_lang_kind: dict[str, Counter] = defaultdict(Counter)
    for target, kind in zip(df["target"], df["kind"]):
        for lang in (json.loads(target).get("category") or {}):
            per_lang_kind[resolve(lang)][kind] += 1

    missing = [l for l in LANGUAGES if l not in per_lang_kind]
    print(f"languages covered = {len(LANGUAGES) - len(missing)}/{len(LANGUAGES)}")
    if missing:
        print(f"  MISSING: {missing}")

    print("\nper-language coverage (single / multi / benign):")
    print(f"  {'language':32} {'single':>7} {'multi':>6} {'benign':>7} {'total':>6}")
    for lang in LANGUAGES:
        c = per_lang_kind[lang]
        tot = sum(c.values())
        flag = "  <-- thin" if tot < 10 else ""
        print(f"  {lang:32} {c['single']:>7} {c['multi']:>6} {c['benign']:>7} {tot:>6}{flag}")

    # independent leak re-check
    train_hashes: set[str] = set()
    for split in ("train", "val"):
        d = pd.read_parquet(V1_SPLITS / f"{split}.parquet", columns=["code"])
        train_hashes.update(norm_hash(c) for c in d["code"])
    for pattern in ACCUKNOX_TRAIN_GLOBS:
        for p in sorted(ACCUKNOX.glob(pattern)):
            d = pd.read_parquet(p, columns=["prompt"])
            train_hashes.update(norm_hash(c) for c in d["prompt"])
    eval_hashes = df["prompt"].map(norm_hash)
    leaked = int(eval_hashes.isin(train_hashes).sum())
    dups = int(eval_hashes.duplicated().sum())
    print(f"\nleak re-check: {leaked} eval rows overlap training sources")
    print(f"internal duplicate prompts: {dups}")

    ok = (not missing) and leaked == 0 and dups == 0
    print("\n" + ("OK: eval set is complete, leak-free, and de-duplicated."
                  if ok else "PROBLEM: see flags above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
