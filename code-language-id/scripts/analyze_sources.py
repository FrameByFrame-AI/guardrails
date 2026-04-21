#!/usr/bin/env python3
"""Analyze raw code-language data sources.

This script reads the downloaded Rosetta Code parquet and GitHub Linguist
taxonomy/samples, then writes small CSV/JSON reports used to decide the first
100-150 language taxonomy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", "", value.lower())


def load_linguist_languages(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_linguist_lookup(languages: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name, meta in languages.items():
        if meta.get("type") != "programming":
            continue
        for alias in [name, *(meta.get("aliases") or [])]:
            lookup.setdefault(normalize_label(alias), name)
    return lookup


def count_linguist_samples(samples_dir: Path) -> pd.DataFrame:
    rows = []
    for language_dir in sorted(samples_dir.iterdir()):
        if not language_dir.is_dir():
            continue
        file_count = sum(1 for path in language_dir.rglob("*") if path.is_file())
        rows.append({"language": language_dir.name, "sample_files": file_count})
    return pd.DataFrame(rows).sort_values(
        ["sample_files", "language"], ascending=[False, True]
    )


def analyze(data_root: Path) -> None:
    rosetta_path = data_root / "raw/rosetta-code/train.parquet"
    linguist_root = data_root / "raw/github-linguist/linguist"
    languages_path = linguist_root / "lib/linguist/languages.yml"
    samples_dir = linguist_root / "samples"
    reports_dir = data_root / "interim/reports"
    label_maps_dir = data_root / "interim/label_maps"

    reports_dir.mkdir(parents=True, exist_ok=True)
    label_maps_dir.mkdir(parents=True, exist_ok=True)

    rosetta = pd.read_parquet(
        rosetta_path,
        columns=["language_name", "task_name", "code"],
    )
    rosetta["code_len"] = rosetta["code"].str.len()

    rosetta_counts = (
        rosetta.groupby("language_name", dropna=False)
        .agg(
            rows=("language_name", "size"),
            tasks=("task_name", "nunique"),
            min_code_len=("code_len", "min"),
            median_code_len=("code_len", "median"),
            mean_code_len=("code_len", "mean"),
            max_code_len=("code_len", "max"),
            empty_code_rows=("code_len", lambda values: int((values == 0).sum())),
        )
        .reset_index()
        .sort_values(["rows", "language_name"], ascending=[False, True])
    )

    languages = load_linguist_languages(languages_path)
    linguist_rows = []
    type_counts: dict[str, int] = {}
    for name, meta in languages.items():
        language_type = meta.get("type", "unknown")
        type_counts[language_type] = type_counts.get(language_type, 0) + 1
        linguist_rows.append(
            {
                "language": name,
                "type": language_type,
                "aliases": "|".join(meta.get("aliases") or []),
                "extensions": "|".join(meta.get("extensions") or []),
            }
        )
    linguist_df = pd.DataFrame(linguist_rows).sort_values(["type", "language"])
    sample_counts = count_linguist_samples(samples_dir)
    linguist_df = linguist_df.merge(
        sample_counts,
        on="language",
        how="left",
    )
    linguist_df["sample_files"] = linguist_df["sample_files"].fillna(0).astype(int)

    lookup = build_linguist_lookup(languages)
    mapping_rows = []
    for row in rosetta_counts.itertuples(index=False):
        raw_label = row.language_name
        canonical = lookup.get(normalize_label(raw_label))
        mapping_rows.append(
            {
                "raw_rosetta_label": raw_label,
                "canonical_language": canonical or "",
                "mapping_type": "auto" if canonical else "unmatched",
                "rows": int(row.rows),
                "tasks": int(row.tasks),
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)

    auto_mapped = mapping_df[mapping_df["canonical_language"] != ""]
    auto_counts = (
        auto_mapped.groupby("canonical_language")
        .agg(
            rosetta_rows=("rows", "sum"),
            rosetta_raw_labels=("raw_rosetta_label", "nunique"),
            rosetta_tasks=("tasks", "sum"),
        )
        .reset_index()
        .merge(
            linguist_df[["language", "sample_files"]],
            left_on="canonical_language",
            right_on="language",
            how="left",
        )
        .drop(columns=["language"])
        .fillna({"sample_files": 0})
    )
    auto_counts["sample_files"] = auto_counts["sample_files"].astype(int)
    auto_counts = auto_counts.sort_values(
        ["rosetta_rows", "canonical_language"], ascending=[False, True]
    )

    unmatched = mapping_df[mapping_df["canonical_language"] == ""].sort_values(
        ["rows", "raw_rosetta_label"], ascending=[False, True]
    )

    rosetta_counts.to_csv(reports_dir / "rosetta_language_counts.csv", index=False)
    linguist_df.to_csv(reports_dir / "linguist_languages.csv", index=False)
    sample_counts.to_csv(reports_dir / "linguist_sample_counts.csv", index=False)
    mapping_df.to_csv(label_maps_dir / "rosetta_to_linguist_auto.csv", index=False)
    auto_counts.to_csv(reports_dir / "auto_mapped_language_counts.csv", index=False)
    unmatched.to_csv(reports_dir / "unmatched_rosetta_labels.csv", index=False)

    summary = {
        "data_root": str(data_root),
        "rosetta_rows": int(len(rosetta)),
        "rosetta_raw_language_labels": int(rosetta_counts.shape[0]),
        "rosetta_empty_code_rows": int((rosetta["code_len"] == 0).sum()),
        "rosetta_code_len_median": float(rosetta["code_len"].median()),
        "rosetta_code_len_mean": float(rosetta["code_len"].mean()),
        "rosetta_code_len_max": int(rosetta["code_len"].max()),
        "linguist_total_languages": int(len(languages)),
        "linguist_by_type": type_counts,
        "linguist_sample_language_dirs": int(sample_counts.shape[0]),
        "linguist_sample_files": int(sample_counts["sample_files"].sum()),
        "auto_mapped_rosetta_raw_labels": int(auto_mapped.shape[0]),
        "auto_mapped_rosetta_rows": int(auto_mapped["rows"].sum()),
        "unmatched_rosetta_raw_labels": int(unmatched.shape[0]),
        "unmatched_rosetta_rows": int(unmatched["rows"].sum()),
        "canonical_auto_mapped_languages": int(auto_counts.shape[0]),
        "canonical_auto_mapped_ge_25_rows": int(
            (auto_counts["rosetta_rows"] >= 25).sum()
        ),
        "canonical_auto_mapped_ge_50_rows": int(
            (auto_counts["rosetta_rows"] >= 50).sum()
        ),
        "canonical_auto_mapped_ge_100_rows": int(
            (auto_counts["rosetta_rows"] >= 100).sum()
        ),
    }
    with (reports_dir / "summary.json").open("w", encoding="utf-8") as handle:
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
    analyze(args.data_root.expanduser())


if __name__ == "__main__":
    main()
