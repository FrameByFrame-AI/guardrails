#!/usr/bin/env python3
"""Assert the unified taxonomy resolves every label seen across all sources.

Sources checked:
  - raw the-stack-v1 parquet slugs (filenames)
  - v1_splits_clean canonical_language + raw_language values
  - Accuknox CodeLanguage category labels (train + test)

Fails loudly (non-zero exit) if any label does not resolve, listing the misses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import (  # noqa: E402
    LANGUAGES,
    NUM_LANGUAGES,
    resolve,
)

MODELS = Path("/models")
STACK_DIR = MODELS / "guardrail_code_data/raw/the-stack-v1"
SPLITS = MODELS / "guardrail_code_data/processed/v1_splits_clean"
ACCUKNOX = MODELS / "Accuknoxtechnologies-CodeLanguage/data"


def main() -> int:
    misses: list[tuple[str, str]] = []
    covered: set[str] = set()

    def check(label: str, source: str) -> None:
        canon = resolve(label, strict=False)
        if canon is None:
            misses.append((source, label))
        else:
            covered.add(canon)

    # 1. stack slugs
    for p in sorted(STACK_DIR.glob("*.parquet")):
        check(p.stem, f"stack-slug:{p.stem}")

    # 2. v1_splits canonical + raw language
    for split in ("train", "val", "test"):
        df = pd.read_parquet(SPLITS / f"{split}.parquet",
                             columns=["raw_language", "canonical_language"])
        for v in df["canonical_language"].unique():
            check(str(v), f"v1.canonical:{split}")
        for v in df["raw_language"].unique():
            check(str(v), f"v1.raw:{split}")

    # 3. Accuknox category labels
    for p in sorted(ACCUKNOX.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["target"]) if "target" in \
            pd.read_parquet(p).columns else None
        if df is None:
            continue
        for t in df["target"]:
            obj = json.loads(t) if isinstance(t, str) else t
            for lang in (obj.get("category") or {}):
                check(lang, f"accuknox:{p.name}")

    print(f"taxonomy size = {NUM_LANGUAGES}")
    print(f"distinct canonical languages covered by data = {len(covered)}")
    uncovered = sorted(set(LANGUAGES) - covered)
    print(f"canonical languages with NO data row = {len(uncovered)}: {uncovered}")

    if misses:
        print(f"\nUNRESOLVED LABELS ({len(misses)}):")
        for source, label in sorted(set(misses)):
            print(f"  [{source}] {label!r}")
        return 1
    print("\nOK: every source label resolves to a canonical language.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
