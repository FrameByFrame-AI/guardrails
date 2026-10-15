#!/usr/bin/env python3
"""Characterize multi-row failures of a multilabel checkpoint on an eval parquet.

Splits multi-row errors into FN (missed a true secondary language) vs FP (added
a spurious language), reports which languages drive each, and how many errors are
"near misses" (the missing label's prob was just under its threshold) — telling
us whether the fix is recall/threshold or genuine model capacity/data.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import LANGUAGES, resolve  # noqa: E402
from eval_multilabel import run_model, checkpoint_thresholds  # noqa: E402

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/models/guardrail_code_models/"
            "programming-language-identification-113-multilabel-v1/model.pt")
PARQUET = Path(sys.argv[2] if len(sys.argv) > 2 else
               "/models/Accuknoxtechnologies-CodeLanguage/data/"
               "eval-unified-113-00000-of-00001.parquet")


def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, labels, isv_true, kinds, texts = run_model(
        str(CKPT), str(PARQUET), device, batch_size=128)
    g, per_class = checkpoint_thresholds(CKPT)
    thr = (np.array([per_class[l] for l in LANGUAGES], dtype=np.float32)
           if per_class else np.full(len(LANGUAGES), g or 0.5, dtype=np.float32))

    multi = kinds == "multi"
    P = probs[multi] >= thr
    T = labels[multi].astype(bool)
    prob_m = probs[multi]

    n = int(multi.sum())
    exact = int((P == T).all(axis=1).sum())
    has_fn = (T & ~P).any(axis=1)
    has_fp = (~T & P).any(axis=1)
    fn_only = int((has_fn & ~has_fp).sum())
    fp_only = int((has_fp & ~has_fn).sum())
    both = int((has_fn & has_fp).sum())

    print(f"multi rows = {n}  exact = {exact} ({100*exact/n:.1f}%)  "
          f"wrong = {n-exact} ({100*(n-exact)/n:.1f}%)")
    print(f"  FN-only (missed a real secondary)  = {fn_only}")
    print(f"  FP-only (added a spurious label)   = {fp_only}")
    print(f"  both                                = {both}")

    # near-miss analysis: for each FN, how far below threshold was it?
    fn_mask = T & ~P
    gaps = []
    for i, j in zip(*np.where(fn_mask)):
        gaps.append(thr[j] - prob_m[i, j])
    gaps = np.array(gaps)
    print(f"\nmissed-label (FN) count = {len(gaps)}")
    if len(gaps):
        print(f"  near-miss (within 0.15 of threshold) = {(gaps < 0.15).sum()} "
              f"({100*(gaps<0.15).mean():.0f}%)")
        print(f"  far-miss  (prob essentially 0)       = {(gaps > 0.5).sum()} "
              f"({100*(gaps>0.5).mean():.0f}%)")

    fn_langs = Counter(LANGUAGES[j] for _, j in zip(*np.where(fn_mask)))
    fp_langs = Counter(LANGUAGES[j] for _, j in zip(*np.where(~T & P)))
    print("\ntop MISSED (FN) languages in multi rows:")
    for l, c in fn_langs.most_common(12):
        print(f"  {l:14} {c}")
    print("\ntop SPURIOUS (FP) languages in multi rows:")
    for l, c in fp_langs.most_common(12):
        print(f"  {l:14} {c}")


if __name__ == "__main__":
    main()
