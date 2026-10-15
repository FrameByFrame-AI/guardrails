#!/usr/bin/env python3
"""Build the v7 "rich" training parquet for the 25-multilabel ByteHybrid.

Sources:
  1. ``train-clean-00000-of-00001.parquet`` (AccuKnox, after our leak-free re-split)
  2. ``stack-polyglot-00000-of-00001.parquet`` (extracted by
     extract_polyglot_from_stack.py; mix of single + multi rows from the six
     polyglot-friendly file types fetched from the-stack-v1)

The fetched/polyglot rows tend to dwarf AccuKnox per-language counts, so we
cap each (kind, primary-language) bucket from the new source. The AccuKnox
rows are kept untouched — they represent the target distribution we evaluate
against and should not be subsampled.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


WEAK_LANGS = ("Bash", "Dockerfile", "Makefile", "SQL", "Terraform", "YAML")


def primary_lang_of(target_str: str) -> str:
    obj = json.loads(target_str) if isinstance(target_str, str) else target_str
    cats = obj.get("category") or {}
    if not cats:
        return "<benign>"
    return sorted(cats.keys())[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accuknox-train", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-clean-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--stack-polyglot", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/stack-polyglot-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--out", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-rich-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--per-bucket-cap", type=int, default=1200,
                   help="Cap rows per (kind, primary-language) bucket from the polyglot source. "
                        "AccuKnox train is never subsampled.")
    p.add_argument("--seed", type=int, default=20260601)
    args = p.parse_args()

    rng = random.Random(args.seed)

    acc = pd.read_parquet(args.accuknox_train)
    print(f"AccuKnox train: {len(acc)} rows")
    print(f"  kind: {dict(Counter(acc['kind']))}")

    if not args.stack_polyglot.exists():
        raise FileNotFoundError(args.stack_polyglot)
    pol = pd.read_parquet(args.stack_polyglot)
    print(f"Polyglot rows: {len(pol)}")
    print(f"  kind: {dict(Counter(pol['kind']))}")

    # Subsample polyglot rows per (kind, primary-language) bucket.
    pol = pol.copy()
    pol["_primary"] = pol["target"].map(primary_lang_of)
    pol["_bucket"] = list(zip(pol["kind"], pol["_primary"]))
    keep_idx: list[int] = []
    bucket_counts: Counter = Counter()
    for bucket, group in pol.groupby("_bucket"):
        idxs = group.index.tolist()
        if len(idxs) > args.per_bucket_cap:
            idxs = rng.sample(idxs, args.per_bucket_cap)
        keep_idx.extend(idxs)
        bucket_counts[bucket] = len(idxs)
    pol_capped = pol.loc[keep_idx, ["prompt", "target", "kind"]].copy()
    print(f"Polyglot after cap (per-bucket={args.per_bucket_cap}): {len(pol_capped)} rows")
    for b, n in sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {b[0]:8s} primary={b[1]:14s} {n:>5}")

    rich = pd.concat([acc[["prompt", "target", "kind"]], pol_capped], ignore_index=True)
    rich = rich.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    final_kind = Counter(rich["kind"])
    final_lang: Counter = Counter()
    for t in rich["target"]:
        obj = json.loads(t) if isinstance(t, str) else t
        for lang in (obj.get("category") or {}):
            final_lang[lang] += 1

    print()
    print(f"FINAL rich train: {len(rich)} rows")
    print(f"  kind: {dict(final_kind)}")
    print("  per-language counts (sum across multi+single):")
    for lang, n in sorted(final_lang.items(), key=lambda x: (-x[1], x[0])):
        marker = " *" if lang in WEAK_LANGS else ""
        print(f"    {lang:14s} {n:>5}{marker}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rich.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
