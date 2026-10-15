#!/usr/bin/env python3
"""Re-split the Accuknox CodeLanguage dataset by code-snippet hash.

The upstream `train`/`test` split has 100% prompt overlap (every test row is a
verbatim duplicate of a training row). This script:

1. Combines all rows from the upstream parquets.
2. Extracts a "code-like" view of each prompt via a heuristic (lines with
   code tokens or high punctuation density). Benign prose rows produce an
   empty code view and get hashed by their prose instead.
3. Groups rows by (code_hash, target) and assigns each group entirely to
   train or test so no code snippet crosses splits.
4. Writes new parquets to OUT_DIR/{train,test}-clean-00000-of-00001.parquet
   with the same schema as upstream (prompt, target, kind).

Default ratio: 90/10. Output kept in the same dataset directory so downstream
code can swap the `--data-dir` arg without surgery.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CODE_TOKENS = re.compile(
    r"[{}\[\]<>;=$#`@|&]"
    r"|\b(def|fn|func|class|import|from|return|if|else|elif|while|for|switch|case|"
    r"var|let|const|public|private|static|void|int|float|double|string|bool|true|"
    r"false|null|nil|None|True|False|self|this|sub|use|require|include|module|"
    r"package|namespace|select|insert|update|delete|where|join|api|kind|metadata|"
    r"spec|resource|provider|version|services|build|run|cmd|copy|add|expose|env|"
    r"workdir|entrypoint|user|volume)\b"
)


def is_code_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 3:
        return False
    if CODE_TOKENS.search(line):
        return True
    special = sum(1 for c in line if not (c.isalpha() or c.isspace() or c in ".,!?'"))
    return special / len(line) > 0.15


def extract_code(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if is_code_line(ln))


def group_key(prompt: str) -> str:
    code = extract_code(prompt)
    # benign / pure-prose rows: hash the prompt itself so two prose rows with
    # identical prose still group, but pure-prose stays out of "shared code".
    payload = code if code else f"prose:{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path("~/llm_models/Accuknoxtechnologies-CodeLanguage/data").expanduser())
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: same as --data-dir (new files use -clean suffix).")
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--prefix", default="clean",
                   help="Filename prefix for outputs: {train,test}-{prefix}-00000-of-00001.parquet.")
    args = p.parse_args()
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(args.data_dir / "train-00000-of-00001.parquet")
    te = pd.read_parquet(args.data_dir / "test-00000-of-00001.parquet")
    df = pd.concat([tr, te], ignore_index=True)
    print(f"upstream: train={len(tr)} test={len(te)} total={len(df)}", flush=True)

    df["group_key"] = df["prompt"].map(group_key)
    n_groups = df["group_key"].nunique()
    print(f"unique code/prose groups: {n_groups} "
          f"({n_groups / len(df) * 100:.1f}% of rows are uniquely-grouped)", flush=True)

    # Build {group_key: [row_indices]} and assign whole groups to train or test
    # in random order until we hit the target test size.
    rng = random.Random(args.seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(df["group_key"].tolist()):
        groups[k].append(i)
    keys = list(groups.keys())
    rng.shuffle(keys)

    target_test = int(len(df) * args.test_ratio)
    test_indices: list[int] = []
    train_indices: list[int] = []
    for k in keys:
        if len(test_indices) < target_test:
            test_indices.extend(groups[k])
        else:
            train_indices.extend(groups[k])

    new_tr = df.iloc[train_indices].drop(columns=["group_key"]).reset_index(drop=True)
    new_te = df.iloc[test_indices].drop(columns=["group_key"]).reset_index(drop=True)
    print(f"new split: train={len(new_tr)} test={len(new_te)} "
          f"(target_test={target_test})", flush=True)

    # Sanity: no group leakage and no exact prompt overlap.
    tr_keys = set(df.iloc[train_indices]["group_key"])
    te_keys = set(df.iloc[test_indices]["group_key"])
    assert not (tr_keys & te_keys), "group_key leakage!"
    overlap = set(new_tr["prompt"]) & set(new_te["prompt"])
    print(f"exact prompt overlap (should be 0): {len(overlap)}", flush=True)

    # Kind distribution check
    print("\nkind distribution:")
    for split_name, split in (("train", new_tr), ("test", new_te)):
        c = Counter(split["kind"])
        total = sum(c.values())
        parts = " ".join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in sorted(c.items()))
        print(f"  {split_name:5s}  {parts}")

    train_out = out_dir / f"train-{args.prefix}-00000-of-00001.parquet"
    test_out = out_dir / f"test-{args.prefix}-00000-of-00001.parquet"
    new_tr.to_parquet(train_out, index=False)
    new_te.to_parquet(test_out, index=False)
    print(f"\nwrote {train_out}")
    print(f"wrote {test_out}")


if __name__ == "__main__":
    main()
