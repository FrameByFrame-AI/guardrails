#!/usr/bin/env python3
"""Create grouped train/val/test splits for the v1 code-language dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()


def stable_float(value: str, seed: int) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def assign_split(group_value: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    score = stable_float(group_value, seed)
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def build(data_root: Path, seed: int, val_ratio: float, test_ratio: float) -> None:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to < 1")

    input_path = data_root / "processed/v1_code_language.parquet"
    split_dir = data_root / "processed/v1_splits"
    reports_dir = data_root / "interim/reports"
    split_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["split"] = df["task_name"].map(
        lambda task: assign_split(str(task), seed, val_ratio, test_ratio)
    )

    split_paths = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        output_path = split_dir / f"{split_name}.parquet"
        split_df.to_parquet(output_path, index=False)
        split_paths[split_name] = str(output_path)

    coverage = (
        df.groupby(["split", "canonical_language"])
        .agg(
            rows=("id", "size"),
            tasks=("task_name", "nunique"),
        )
        .reset_index()
        .sort_values(["canonical_language", "split"])
    )
    coverage.to_csv(reports_dir / "v1_split_language_coverage.csv", index=False)

    split_summary = (
        df.groupby("split")
        .agg(
            rows=("id", "size"),
            tasks=("task_name", "nunique"),
            languages=("canonical_language", "nunique"),
        )
        .reset_index()
        .sort_values("split")
    )
    split_summary.to_csv(reports_dir / "v1_split_summary.csv", index=False)

    task_splits = df.groupby("task_name")["split"].nunique()
    leaked_tasks = task_splits[task_splits > 1]
    label_split_counts = coverage.pivot_table(
        index="canonical_language",
        columns="split",
        values="rows",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for split_name in ["train", "val", "test"]:
        if split_name not in label_split_counts:
            label_split_counts[split_name] = 0
    label_split_counts["missing_from_val"] = label_split_counts["val"] == 0
    label_split_counts["missing_from_test"] = label_split_counts["test"] == 0
    label_split_counts.to_csv(
        reports_dir / "v1_split_label_gaps.csv",
        index=False,
    )

    summary = {
        "input_path": str(input_path),
        "split_paths": split_paths,
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "rows": int(len(df)),
        "tasks": int(df["task_name"].nunique()),
        "languages": int(df["canonical_language"].nunique()),
        "split_rows": {
            row.split: int(row.rows) for row in split_summary.itertuples(index=False)
        },
        "split_tasks": {
            row.split: int(row.tasks) for row in split_summary.itertuples(index=False)
        },
        "task_leakage_count": int(len(leaked_tasks)),
        "labels_missing_from_val": int(label_split_counts["missing_from_val"].sum()),
        "labels_missing_from_test": int(label_split_counts["missing_from_test"].sum()),
    }
    with (reports_dir / "v1_split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing processed/ dataset folders.",
    )
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    build(args.data_root.expanduser(), args.seed, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
