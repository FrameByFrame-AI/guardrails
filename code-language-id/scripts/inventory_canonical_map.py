#!/usr/bin/env python3
"""Extract the authoritative raw->canonical language map from v1_splits_clean.

The lite model's labels live in the `canonical_language` column; `raw_language`
and `source` show where each came from. This dumps:
  - distinct canonical_language values (the 107) + counts
  - raw_language -> canonical_language mapping (per source)
  - stack parquet filename -> canonical (by matching raw_language)
so we can build the alias map without guessing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SPLITS = Path("/models/guardrail_code_data/processed/v1_splits_clean")


def main() -> None:
    df = pd.concat(
        [pd.read_parquet(SPLITS / f"{s}.parquet") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    print(f"total rows={len(df)}")

    canon = Counter(df["canonical_language"].tolist())
    print(f"\n=== distinct canonical_language = {len(canon)} ===")
    for lang, n in sorted(canon.items()):
        print(f"  {lang!r:40} {n}")

    print("\n=== raw_language -> canonical_language (where they differ) ===")
    pairs = defaultdict(Counter)
    for raw, canon_l, src in zip(df["raw_language"], df["canonical_language"], df["source"]):
        pairs[(raw, canon_l)][src] += 1
    seen = set()
    for (raw, canon_l), srcs in sorted(pairs.items()):
        if str(raw) != str(canon_l):
            key = (str(raw), str(canon_l))
            if key in seen:
                continue
            seen.add(key)
            print(f"  {raw!r:35} -> {canon_l!r:35} sources={dict(srcs)}")

    print("\n=== source distribution ===")
    print(dict(Counter(df["source"].tolist())))


if __name__ == "__main__":
    main()
