#!/usr/bin/env python3
"""Inventory every label source feeding the unified multi-label taxonomy.

Reports, for each source, the exact label strings and row counts so we can
build an accurate canonical taxonomy + alias map:

  - lite 107-class model      : lang2idx.json
  - v1_splits_clean parquets  : the labels the lite model actually trained on
  - raw the-stack-v1 parquets : per-language file dumps (incl. config formats)
  - Accuknox CodeLanguage     : 25-label multi-label schema (train/test)

Run inside any docker-compose service that mounts ~/llm_models at /models.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

MODELS = Path("/models")
LITE_DIR = MODELS / "guardrail_code_models/programming-language-identification-100plus-lite"
SPLITS_DIR = MODELS / "guardrail_code_data/processed/v1_splits_clean"
STACK_DIR = MODELS / "guardrail_code_data/raw/the-stack-v1"
ROSETTA_DIR = MODELS / "guardrail_code_data/raw/rosetta-code"
LINGUIST_DIR = MODELS / "guardrail_code_data/raw/github-linguist"
ACCUKNOX_DIR = MODELS / "Accuknoxtechnologies-CodeLanguage/data"


def sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def label_column(df: pd.DataFrame) -> str | None:
    for cand in ("language", "label", "lang", "languages", "target"):
        if cand in df.columns:
            return cand
    return None


def main() -> None:
    # --- 1. lite 107 label map -------------------------------------------------
    sep("1. LITE 107 label map (lang2idx.json)")
    lite = json.load(open(LITE_DIR / "lang2idx.json"))
    lite_langs = sorted(lite, key=lambda k: lite[k]) if isinstance(lite, dict) else sorted(lite)
    print(f"count={len(lite_langs)}")
    print(lite_langs)

    # --- 2. v1_splits_clean actual label distribution --------------------------
    sep("2. v1_splits_clean label distribution (what lite actually trained on)")
    for split in ("train", "val", "test"):
        path = SPLITS_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"  {split}: MISSING")
            continue
        df = pd.read_parquet(path)
        col = label_column(df)
        print(f"  {split}: rows={len(df)} cols={list(df.columns)} label_col={col!r}")
        if split == "train" and col:
            counts = Counter(df[col].tolist())
            print(f"    distinct labels={len(counts)}")
            for lang, n in sorted(counts.items()):
                print(f"      {lang!r:40} {n}")

    # --- 3. raw the-stack-v1 parquet names + row counts ------------------------
    sep("3. raw the-stack-v1 parquet files (filename -> rows)")
    stack_files = sorted(STACK_DIR.glob("*.parquet"))
    print(f"count={len(stack_files)}")
    for p in stack_files:
        try:
            n = len(pd.read_parquet(p, columns=None))
        except Exception as exc:  # noqa: BLE001
            n = f"ERR:{exc}"
        print(f"  {p.stem:40} {n}")

    # --- 4. rosetta / linguist extras ------------------------------------------
    for name, d in (("rosetta-code", ROSETTA_DIR), ("github-linguist", LINGUIST_DIR)):
        sep(f"4. raw {name} contents")
        if not d.exists():
            print("  (missing)")
            continue
        for p in sorted(d.iterdir()):
            print(f"  {p.name}")

    # --- 5. Accuknox 25-label schema -------------------------------------------
    sep("5. Accuknox CodeLanguage label universe (train/test parquets)")
    for path in sorted(ACCUKNOX_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name}: ERR {exc}")
            continue
        langs: Counter = Counter()
        kinds: Counter = Counter()
        if "target" in df.columns:
            for t in df["target"]:
                obj = json.loads(t) if isinstance(t, str) else t
                for lang, on in (obj.get("category") or {}).items():
                    if on:
                        langs[lang] += 1
        if "kind" in df.columns:
            kinds = Counter(df["kind"].tolist())
        print(f"  {path.name}: rows={len(df)} kinds={dict(kinds)} distinct_langs={len(langs)}")
        if langs:
            print("    " + ", ".join(sorted(langs)))


if __name__ == "__main__":
    main()
