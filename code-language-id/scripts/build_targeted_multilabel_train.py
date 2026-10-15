#!/usr/bin/env python3
"""Build a targeted v8 training parquet for CodeLanguage multilabel.

Starts from the v7 rich train split, then adds:

* pair-focused synthetic multi rows for weak/error-prone pairs
* hard-negative repeats for config/build files that contain shell-like text but
  are labeled as a single non-Bash language

The goal is to improve exact-match on multi rows without changing the base
ByteHybrid architecture.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


TARGET_PAIRS: tuple[tuple[str, str], ...] = (
    ("Scala", "jq"),
    ("Go", "PowerShell"),
    ("C", "Dockerfile"),
    ("Swift", "Terraform"),
    ("Makefile", "Bash"),
    ("Ruby", "jq"),
    ("JavaScript", "YAML"),
    ("PowerShell", "YAML"),
    ("Bash", "Terraform"),
    ("Bash", "Dockerfile"),
)

SHELLISH_TOKENS = (
    "&&", "$(", "${", " bash ", " sh ", "echo ", "curl ", "wget ", "chmod ",
    "mkdir ", "rm ", "cp ", "mv ", "grep ", "sed ", "awk ", "npm ", "pip ",
    "docker ", "make ",
)

SEPARATORS = (
    "\n\nAlso inspect this second snippet:\n",
    "\n\nThe same prompt includes another file:\n",
    "\n\n--- next snippet ---\n",
    "\n\nIn a different part of the repository:\n",
    "\n\nA related example follows:\n",
)


def parse_target(raw: str | dict) -> dict:
    return json.loads(raw) if isinstance(raw, str) else raw


def langs_of(raw: str | dict) -> frozenset[str]:
    obj = parse_target(raw)
    return frozenset((obj.get("category") or {}).keys())


def target_for(langs: set[str] | frozenset[str]) -> str:
    payload = {"is_valid": True, "category": {lang: True for lang in sorted(langs)}}
    return json.dumps(payload, separators=(",", ":"))


def shellish(text: str) -> bool:
    low = f" {text.lower()} "
    return any(tok in low for tok in SHELLISH_TOKENS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-train", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-rich-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--single-source", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-clean-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--out", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-targeted-v8-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--rows-per-pair", type=int, default=160)
    p.add_argument("--hard-negative-repeats", type=int, default=2)
    p.add_argument("--hard-negative-cap", type=int, default=800)
    p.add_argument("--seed", type=int, default=20260602)
    args = p.parse_args()

    rng = random.Random(args.seed)
    base = pd.read_parquet(args.base_train)[["prompt", "target", "kind"]].copy()
    single_source = pd.read_parquet(args.single_source)[["prompt", "target", "kind"]].copy()

    single_source["_langs"] = single_source["target"].map(langs_of)
    singles_by_lang: dict[str, pd.DataFrame] = {}
    for lang in sorted({lang for langs in single_source["_langs"] for lang in langs}):
        singles_by_lang[lang] = single_source[
            (single_source["kind"] == "single") & (single_source["_langs"] == frozenset([lang]))
        ].reset_index(drop=True)

    new_rows: list[dict[str, str]] = []
    for lang_a, lang_b in TARGET_PAIRS:
        src_a = singles_by_lang.get(lang_a)
        src_b = singles_by_lang.get(lang_b)
        if src_a is None or src_b is None or src_a.empty or src_b.empty:
            print(f"[skip] pair {lang_a}+{lang_b}: missing single source rows", flush=True)
            continue
        for _ in range(args.rows_per_pair):
            row_a = src_a.iloc[rng.randrange(len(src_a))]
            row_b = src_b.iloc[rng.randrange(len(src_b))]
            if rng.random() < 0.5:
                first, second = row_a, row_b
            else:
                first, second = row_b, row_a
            prompt = str(first["prompt"]) + rng.choice(SEPARATORS) + str(second["prompt"])
            langs = langs_of(first["target"]) | langs_of(second["target"])
            new_rows.append({
                "prompt": prompt,
                "target": target_for(langs),
                "kind": "multi",
            })
        print(f"added {args.rows_per_pair} targeted rows for {lang_a}+{lang_b}", flush=True)

    base["_langs"] = base["target"].map(langs_of)
    hard_negative_langs = {"Dockerfile", "Makefile", "Terraform", "YAML", "SQL"}
    hard_negatives = base[
        (base["kind"] == "single")
        & base["_langs"].map(lambda langs: "Bash" not in langs and bool(langs & hard_negative_langs))
        & base["prompt"].map(lambda text: shellish(str(text)))
    ][["prompt", "target", "kind"]]
    if len(hard_negatives) > args.hard_negative_cap:
        hard_negatives = hard_negatives.sample(n=args.hard_negative_cap, random_state=args.seed)
    repeated_hard = pd.concat(
        [hard_negatives] * max(0, args.hard_negative_repeats),
        ignore_index=True,
    )
    print(
        f"hard negatives: selected={len(hard_negatives)} "
        f"repeats={args.hard_negative_repeats} emitted={len(repeated_hard)}",
        flush=True,
    )

    targeted = pd.DataFrame(new_rows)
    out = pd.concat(
        [
            base.drop(columns=["_langs"]),
            targeted,
            repeated_hard,
        ],
        ignore_index=True,
    )
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    kind_counts = Counter(out["kind"])
    lang_counts: Counter[str] = Counter()
    for target in out["target"]:
        for lang in langs_of(target):
            lang_counts[lang] += 1

    print()
    print(f"output rows: {len(out)}")
    print(f"kind: {dict(kind_counts)}")
    print("top labels:")
    for lang, count in lang_counts.most_common(12):
        print(f"  {lang:14s} {count:>5}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
