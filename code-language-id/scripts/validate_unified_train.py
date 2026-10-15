#!/usr/bin/env python3
"""Independent validation of the unified 113-lang train + dev corpus.

Checks (from the written parquets):
  - train / dev / 113-eval are pairwise disjoint by normalized prompt hash
  - every language appears in train, and how often as a *secondary* (multi) label
  - kind + is_valid distribution; real-vs-synthetic multi composition
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import LANGUAGES  # noqa: E402
from build_unified_train import ACCUKNOX, EVAL_PARQUET, norm_hash, primary_langs  # noqa: E402

TRAIN = ACCUKNOX / "train-unified-113-00000-of-00001.parquet"
DEV = ACCUKNOX / "dev-unified-113-00000-of-00001.parquet"


def hashes(p: Path) -> set[str]:
    return set(pd.read_parquet(p, columns=["prompt"])["prompt"].map(norm_hash))


def main() -> int:
    train = pd.read_parquet(TRAIN)
    dev = pd.read_parquet(DEV)
    th, dh, eh = hashes(TRAIN), hashes(DEV), hashes(EVAL_PARQUET)

    print(f"train={len(train):,}  dev={len(dev):,}  eval={len(eh):,}")
    print("\ndisjointness (must all be 0):")
    print(f"  train ∩ dev  = {len(th & dh)}")
    print(f"  train ∩ eval = {len(th & eh)}")
    print(f"  dev   ∩ eval = {len(dh & eh)}")

    print(f"\ntrain kinds   = {dict(Counter(train['kind']))}")
    print(f"train sources = {dict(Counter(train['source']))}")

    # multi composition: real vs synthetic
    multi = train[train["kind"] == "multi"]
    real_multi = multi[~multi["source"].str.startswith("synth")]
    print(f"\nmulti rows = {len(multi):,}  (real={len(real_multi):,}, "
          f"synth={len(multi)-len(real_multi):,})")

    # per-language: total appearances and appearances as a SECONDARY in multi
    total_cov, multi_cov = Counter(), Counter()
    for t, k in zip(train["target"], train["kind"]):
        langs = primary_langs(t)
        for l in langs:
            total_cov[l] += 1
            if k == "multi":
                multi_cov[l] += 1
    missing = [l for l in LANGUAGES if total_cov[l] == 0]
    no_multi = [l for l in LANGUAGES if multi_cov[l] == 0]
    print(f"\nlanguages in train       = {len(LANGUAGES)-len(missing)}/{len(LANGUAGES)}")
    print(f"languages w/ multi label = {len(LANGUAGES)-len(no_multi)}/{len(LANGUAGES)}")
    if missing:
        print(f"  MISSING entirely: {missing}")
    if no_multi:
        print(f"  never in a multi row: {no_multi}")

    thin_multi = sorted((multi_cov[l], l) for l in LANGUAGES if multi_cov[l] < 30)
    print(f"\nlanguages with <30 multi rows ({len(thin_multi)}):")
    print("  " + ", ".join(f"{l}={n}" for n, l in thin_multi))

    # span integrity (synthetic multi rows carry byte-offset code spans) --------
    span_bad = span_over = span_mismatch = span_tot = prompt_over = 0
    if "spans" in train.columns:
        synth = train[train["source"].str.startswith("synth")]
        for prompt, target, raw in zip(synth["prompt"], synth["target"], synth["spans"]):
            blen = len(str(prompt).encode("utf-8", "replace"))
            if blen > 2048:
                prompt_over += 1
            cats = set(json.loads(target).get("category") or {})
            for s in json.loads(raw):
                span_tot += 1
                if not (0 <= s["start"] < s["end"] <= blen):
                    span_bad += 1
                if s["end"] > 2048:
                    span_over += 1
                if s["lang"] not in cats:
                    span_mismatch += 1
        print(f"\nspans (synth multi): total={span_tot}  out-of-range={span_bad}  "
              f"end>2048={span_over}  lang-not-in-target={span_mismatch}")
        print(f"synth prompts >2048 bytes = {prompt_over}")
    spans_ok = span_bad == 0 and span_mismatch == 0 and prompt_over == 0

    ok = ((not missing) and spans_ok
          and len(th & dh) == 0 and len(th & eh) == 0 and len(dh & eh) == 0)
    print("\n" + ("OK: corpus is disjoint and fully covered."
                  if ok else "PROBLEM: see flags above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
