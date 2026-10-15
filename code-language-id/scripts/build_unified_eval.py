#!/usr/bin/env python3
"""Build the leak-free 113-language multi-label evaluation set.

The current Accuknox `test-clean` parquet only scores 25 of the 113 unified
languages, so there is no way to measure the long tail. This assembles one
unified eval parquet in the Accuknox schema (`prompt`, `target` JSON, `kind`)
covering all 113:

  single  : held-out v1_splits_clean/test.parquet (107 langs, pure-code) +
            Accuknox test-clean single rows (the 6 config formats + 25 langs)
  multi   : Accuknox test-clean multi rows (gold polyglot)
  benign  : Accuknox test-clean benign rows

Leak safety
-----------
Both inputs are already held out from their own training splits by construction
(v1 group-split; Accuknox `clean` resplit). This script additionally hashes
every prompt against ALL train sources (v1 train/val + every Accuknox train-*
parquet) and drops any eval row that collides, reporting the count. Per-language
single rows from v1 are capped (stable subsample) so common languages don't
dominate the macro metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import LANGUAGES, resolve  # noqa: E402

MODELS = Path("/models")
V1_SPLITS = MODELS / "guardrail_code_data/processed/v1_splits_clean"
ACCUKNOX = MODELS / "Accuknoxtechnologies-CodeLanguage/data"

# The ACTUAL training sources the unified model will use (leak guard).
# Deliberately excludes the deprecated pre-resplit `train-00000`/`test-00000`
# parquets — test-clean is held out from train-CLEAN, not from those.
ACCUKNOX_TRAIN_GLOBS = ("train-clean-00000-of-00001.parquet", "stack-polyglot-*.parquet")


def norm_hash(text: str) -> str:
    """Whitespace-normalized prompt hash for cross-source leak detection."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    collapsed = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(collapsed.encode("utf-8", "replace")).hexdigest()


def single_target(canon: str) -> str:
    return json.dumps({"is_valid": True, "category": {canon: True}}, ensure_ascii=False)


def collect_train_hashes() -> dict[str, set[str]]:
    """Return {source_name: set_of_prompt_hashes} for per-source leak reporting."""
    per_source: dict[str, set[str]] = {}
    # v1 single-label train + val (prompt == code column)
    for split in ("train", "val"):
        df = pd.read_parquet(V1_SPLITS / f"{split}.parquet", columns=["code"])
        per_source[f"v1_{split}"] = {norm_hash(c) for c in df["code"]}
    # genuine Accuknox train sources (prompt column)
    for pattern in ACCUKNOX_TRAIN_GLOBS:
        for p in sorted(ACCUKNOX.glob(pattern)):
            df = pd.read_parquet(p, columns=["prompt"])
            per_source[p.name] = {norm_hash(c) for c in df["prompt"]}
    return per_source


def build_v1_single(max_per_lang: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(
        V1_SPLITS / "test.parquet", columns=["code", "canonical_language"]
    )
    df = df.rename(columns={"code": "prompt"})
    df["canon"] = df["canonical_language"].map(lambda x: resolve(x))
    # stable per-language subsample
    parts = [
        g.sample(n=min(len(g), max_per_lang), random_state=seed)
        for _, g in df.groupby("canon")
    ]
    out = pd.concat(parts, ignore_index=True)
    out["target"] = out["canon"].map(single_target)
    out["kind"] = "single"
    out["source"] = "v1_test"
    return out[["prompt", "target", "kind", "source"]]


def build_accuknox_test() -> pd.DataFrame:
    df = pd.read_parquet(ACCUKNOX / "test-clean-00000-of-00001.parquet")
    # re-resolve category keys through the canonical taxonomy (identity for the 25)
    fixed = []
    for t in df["target"]:
        obj = json.loads(t) if isinstance(t, str) else t
        cats = {resolve(k): True for k in (obj.get("category") or {})}
        fixed.append(json.dumps(
            {"is_valid": bool(obj.get("is_valid", bool(cats))), "category": cats},
            ensure_ascii=False,
        ))
    df = df.copy()
    df["target"] = fixed
    df["source"] = "accuknox_test_clean"
    return df[["prompt", "target", "kind", "source"]]


def primary_langs(target: str) -> list[str]:
    obj = json.loads(target)
    return sorted(obj.get("category") or {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ACCUKNOX / "eval-unified-113-00000-of-00001.parquet")
    ap.add_argument("--max-single-per-lang", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    print("collecting train-prompt hashes for leak check ...")
    per_source = collect_train_hashes()
    for name, hs in per_source.items():
        print(f"  {name:42} {len(hs):,} hashes")
    all_train = set().union(*per_source.values())

    v1 = build_v1_single(args.max_single_per_lang, args.seed)
    acc = build_accuknox_test()
    print(f"v1 single rows           = {len(v1):,}")
    print(f"accuknox test-clean rows = {len(acc):,}")

    combined = pd.concat([v1, acc], ignore_index=True)
    combined["h"] = combined["prompt"].map(norm_hash)

    # per-source collision report (which train source each eval source overlaps)
    print("\nper-source collisions (eval_source x train_source):")
    for esrc in combined["source"].unique():
        ehashes = set(combined.loc[combined["source"] == esrc, "h"])
        for tname, ths in per_source.items():
            n = len(ehashes & ths)
            if n:
                print(f"  {esrc:22} x {tname:42} = {n}")

    # drop eval rows that collide with any genuine train prompt
    leaked = combined["h"].isin(all_train)
    print(f"\nleaked rows dropped       = {int(leaked.sum()):,}")
    combined = combined[~leaked].copy()

    # drop internal duplicates (keep first)
    before = len(combined)
    combined = combined.drop_duplicates(subset="h", keep="first")
    print(f"internal dup rows dropped = {before - len(combined):,}")
    combined = combined.drop(columns="h").reset_index(drop=True)

    # coverage report
    kinds = Counter(combined["kind"])
    lang_cov: Counter = Counter()
    for t in combined["target"]:
        for lang in primary_langs(t):
            lang_cov[lang] += 1
    missing = [l for l in LANGUAGES if lang_cov[l] == 0]

    print(f"\nfinal eval rows = {len(combined):,}  kinds={dict(kinds)}")
    print(f"languages covered = {len(LANGUAGES) - len(missing)}/{len(LANGUAGES)}")
    if missing:
        print(f"MISSING languages ({len(missing)}): {missing}")
    thin = sorted((n, l) for l, n in lang_cov.items() if n < 10)
    if thin:
        print(f"thin (<10 rows) languages: {[(l, n) for n, l in thin]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
