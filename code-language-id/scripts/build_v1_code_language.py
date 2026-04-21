#!/usr/bin/env python3
"""Build the v1 programming-language dataset from mapped Rosetta rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()


REQUIRED_COLUMNS = [
    "id",
    "source",
    "raw_language",
    "canonical_language",
    "task_name",
    "task_url",
    "language_url",
    "code",
    "code_len",
]


def load_mapped_sources(
    data_root: Path,
    extra_dirs: list[Path],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    rosetta_path = data_root / "interim/normalized/rosetta_mapped.parquet"
    if rosetta_path.exists():
        frames.append(pd.read_parquet(rosetta_path))

    for extra_dir in extra_dirs:
        if not extra_dir.exists():
            continue
        for parquet in sorted(extra_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(parquet))

    if not frames:
        raise FileNotFoundError("no mapped source parquets found")

    combined = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(f"combined frame missing required columns: {missing}")
    return combined[REQUIRED_COLUMNS]


def build(
    data_root: Path,
    min_rows: int,
    extra_dirs: list[Path],
) -> None:
    reports_dir = data_root / "interim/reports"
    processed_dir = data_root / "processed"
    normalized_dir = data_root / "interim/normalized"

    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    combined = load_mapped_sources(data_root, extra_dirs)
    combined["code_len"] = combined["code"].str.len()
    combined = combined[combined["code_len"] > 0].copy()
    combined_path = normalized_dir / "combined_mapped.parquet"
    combined.to_parquet(combined_path, index=False)

    coverage = (
        combined.groupby("canonical_language")
        .agg(
            rows=("id", "size"),
            raw_labels=("raw_language", "nunique"),
            tasks=("task_name", "nunique"),
            median_code_len=("code_len", "median"),
            mean_code_len=("code_len", "mean"),
            max_code_len=("code_len", "max"),
        )
        .reset_index()
        .sort_values(["rows", "canonical_language"], ascending=[False, True])
    )
    coverage.to_csv(reports_dir / "combined_mapped_coverage.csv", index=False)

    selected = coverage[coverage["rows"] >= min_rows].copy()
    selected = selected.sort_values(["rows", "canonical_language"], ascending=[False, True])
    selected.insert(0, "label_id", range(len(selected)))
    selected["include"] = True
    selected["selection_rule"] = f"combined_rows >= {min_rows}"
    selected["notes"] = ""

    languages_path = reports_dir / "v1_languages.csv"
    selected.to_csv(languages_path, index=False)

    selected_languages = set(selected["canonical_language"])
    v1 = combined[combined["canonical_language"].isin(selected_languages)].copy()
    label_ids = dict(zip(selected["canonical_language"], selected["label_id"]))
    v1["language_label_id"] = v1["canonical_language"].map(label_ids).astype(int)

    output_path = processed_dir / "v1_code_language.parquet"
    v1.to_parquet(output_path, index=False)

    v1_coverage = (
        v1.groupby(["language_label_id", "canonical_language"])
        .agg(
            rows=("id", "size"),
            raw_labels=("raw_language", "nunique"),
            tasks=("task_name", "nunique"),
            median_code_len=("code_len", "median"),
            mean_code_len=("code_len", "mean"),
            max_code_len=("code_len", "max"),
        )
        .reset_index()
        .sort_values(["rows", "canonical_language"], ascending=[False, True])
    )
    v1_coverage.to_csv(reports_dir / "v1_code_language_coverage.csv", index=False)

    summary = {
        "min_rows": int(min_rows),
        "languages_path": str(languages_path),
        "output_path": str(output_path),
        "languages": int(selected.shape[0]),
        "rows": int(v1.shape[0]),
        "raw_language_labels": int(v1["raw_language"].nunique()),
        "tasks": int(v1["task_name"].nunique()),
        "code_len_median": float(v1["code_len"].median()),
        "code_len_mean": float(v1["code_len"].mean()),
        "code_len_max": int(v1["code_len"].max()),
    }
    with (reports_dir / "v1_code_language_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing interim/ and processed/ dataset folders.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Minimum mapped rows across all sources required to include a language.",
    )
    parser.add_argument(
        "--extra-source-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory containing additional source parquets to concatenate. "
            "May be passed multiple times. Defaults to raw/the-stack-v1 if present."
        ),
    )
    args = parser.parse_args()
    extra_dirs: list[Path] = args.extra_source_dir or []
    if not extra_dirs:
        default_extra = args.data_root.expanduser() / "raw/the-stack-v1"
        if default_extra.exists():
            extra_dirs = [default_extra]
    build(args.data_root.expanduser(), args.min_rows, extra_dirs)


if __name__ == "__main__":
    main()
