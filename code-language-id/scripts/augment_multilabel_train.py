#!/usr/bin/env python3
"""Augment the multi-label train split with synthetic multi-language rows.

The Accuknox CodeLanguage train split has a 70/20/10 split of
single/multi/benign — the 25-multi-label model's main error mode is missing
a secondary language in `multi` prompts (~41% error rate on those rows).

This script generates synthetic multi rows by sampling 2-3 single-language
rows with *different* languages from the training split, concatenating their
prompts with a small natural separator, and unioning their labels. We never
touch the test split. Pair-IDs are tracked to avoid duplicates.

Default: emit enough synthetic rows to bring multi% from ~20% to ~35% of the
augmented training set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


SEPARATORS = (
    "\n\nAlso, here's another snippet:\n",
    "\n\nAnd separately:\n",
    "\n\nIn addition:\n",
    "\n\n---\n\n",
    "\n\nLater in the same file:\n",
    "\n\nFor comparison, consider:\n",
)


def parse_target(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


def union_targets(targets: list[dict]) -> dict:
    cats: dict[str, bool] = {}
    for t in targets:
        for lang in (t.get("category") or {}):
            cats[lang] = True
    return {"is_valid": True, "category": cats}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-clean-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--out-parquet", type=Path, default=Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/train-clean-aug-00000-of-00001.parquet"
    ).expanduser())
    p.add_argument("--num-2way", type=int, default=2000,
                   help="Number of synthetic 2-language rows to generate.")
    p.add_argument("--num-3way", type=int, default=500,
                   help="Number of synthetic 3-language rows to generate.")
    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--max-attempts-per-row", type=int, default=20,
                   help="Cap on resampling attempts when language overlap is bad.")
    args = p.parse_args()

    df = pd.read_parquet(args.train_parquet)
    df["target_obj"] = df["target"].map(parse_target)
    df["langs"] = df["target_obj"].map(lambda t: frozenset((t.get("category") or {}).keys()))

    # Only sample from single-language rows to keep label semantics clean.
    singles = df[df["kind"] == "single"].reset_index(drop=True)
    print(f"loaded {len(df)} train rows ({len(singles)} singles available)", flush=True)

    rng = random.Random(args.seed)
    seen_combos: set[tuple] = set()
    new_rows: list[dict] = []

    def sample_combo(n_langs: int) -> tuple[list[int], frozenset[str]] | None:
        for _ in range(args.max_attempts_per_row):
            idxs = rng.sample(range(len(singles)), n_langs)
            langs_each = [next(iter(singles.iloc[i]["langs"])) for i in idxs]
            langs = frozenset(langs_each)
            if len(langs) != n_langs:
                continue  # collision — same language sampled twice
            key = (n_langs,) + tuple(sorted(zip(idxs, langs_each)))
            if key in seen_combos:
                continue
            seen_combos.add(key)
            return idxs, langs
        return None

    for n_langs, n_target in ((2, args.num_2way), (3, args.num_3way)):
        emitted = 0
        attempts = 0
        max_attempts = n_target * 5
        while emitted < n_target and attempts < max_attempts:
            attempts += 1
            picked = sample_combo(n_langs)
            if picked is None:
                continue
            idxs, langs = picked
            rows = [singles.iloc[i] for i in idxs]
            prompts = [r["prompt"] for r in rows]
            targets = [r["target_obj"] for r in rows]
            sep = rng.choice(SEPARATORS)
            combined_prompt = sep.join(prompts)
            combined_target = union_targets(targets)
            new_rows.append({
                "prompt": combined_prompt,
                "target": json.dumps(combined_target, separators=(",", ":")),
                "kind": "multi",
            })
            emitted += 1
        print(f"  emitted {emitted}/{n_target} synthetic {n_langs}-language rows "
              f"(took {attempts} attempts)", flush=True)

    aug = pd.DataFrame(new_rows)
    # Drop our scratch columns before concat (they don't exist on aug)
    out = pd.concat(
        [df.drop(columns=["target_obj", "langs"]), aug],
        ignore_index=True,
    )

    print("\naugmented train kind distribution:")
    c = Counter(out["kind"])
    total = sum(c.values())
    for k, v in sorted(c.items()):
        print(f"  {k:8s}  {v:>6}  ({v/total*100:.1f}%)")
    print(f"  total     {total:>6}")

    # Re-shuffle so synthetic rows are mixed in
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_parquet(args.out_parquet, index=False)
    print(f"\nwrote {args.out_parquet}")


if __name__ == "__main__":
    main()
