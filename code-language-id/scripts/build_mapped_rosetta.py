#!/usr/bin/env python3
"""Build a matched-only Rosetta Code dataset.

Rows whose Rosetta labels do not automatically map to a GitHub Linguist
programming language are excluded for the first dataset version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()


def build(data_root: Path) -> None:
    rosetta_path = data_root / "raw/rosetta-code/train.parquet"
    auto_map_path = data_root / "interim/label_maps/rosetta_to_linguist_auto.csv"
    manual_map_path = data_root / "interim/label_maps/rosetta_manual_overrides.csv"
    normalized_dir = data_root / "interim/normalized"
    reports_dir = data_root / "interim/reports"

    normalized_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rosetta = pd.read_parquet(rosetta_path)
    auto_map = pd.read_csv(auto_map_path).fillna("")
    auto_map = auto_map[auto_map["canonical_language"] != ""]
    label_map = dict(
        zip(auto_map["raw_rosetta_label"], auto_map["canonical_language"])
    )

    if manual_map_path.exists():
        manual_map = pd.read_csv(manual_map_path).fillna("")
        manual_map = manual_map[manual_map["canonical_language"] != ""]
        for raw_label, canonical in zip(
            manual_map["raw_rosetta_label"], manual_map["canonical_language"]
        ):
            label_map[raw_label] = canonical

    mapped = rosetta[rosetta["language_name"].isin(label_map)].copy()
    mapped["canonical_language"] = mapped["language_name"].map(label_map)
    mapped["source"] = "rosetta-code"
    mapped["code_len"] = mapped["code"].str.len()
    mapped = mapped[mapped["code_len"] > 0].copy()

    mapped = mapped.rename(
        columns={
            "language_name": "raw_language",
        }
    )
    mapped.insert(0, "id", [f"rosetta-{idx}" for idx in mapped.index])

    columns = [
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
    mapped = mapped[columns]

    output_path = normalized_dir / "rosetta_mapped.parquet"
    mapped.to_parquet(output_path, index=False)

    coverage = (
        mapped.groupby("canonical_language")
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
    coverage.to_csv(reports_dir / "rosetta_mapped_coverage.csv", index=False)

    summary = {
        "output_path": str(output_path),
        "rows": int(len(mapped)),
        "canonical_languages": int(mapped["canonical_language"].nunique()),
        "raw_language_labels": int(mapped["raw_language"].nunique()),
        "dropped_unmatched_or_empty_rows": int(len(rosetta) - len(mapped)),
        "languages_ge_25_rows": int((coverage["rows"] >= 25).sum()),
        "languages_ge_50_rows": int((coverage["rows"] >= 50).sum()),
        "languages_ge_100_rows": int((coverage["rows"] >= 100).sum()),
    }
    with (reports_dir / "rosetta_mapped_summary.json").open(
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
        help="Root directory containing raw/ and interim/ dataset folders.",
    )
    args = parser.parse_args()
    build(args.data_root.expanduser())


if __name__ == "__main__":
    main()
