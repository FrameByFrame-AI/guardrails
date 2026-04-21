#!/usr/bin/env python3
"""Drop Tier A mismatches (high-confidence wrong labels) from the v1 splits.

Tier A definition:
  A canonical_language qualifies as "mainstream" when the full-dataset LLM
  validation match rate is >= --threshold (default 0.95). A row is a Tier A
  mismatch when both its canonical_language and its predicted_language
  (after alias resolution) are mainstream AND the two differ.

Produces filtered train/val/test parquets in --output-dir (mirroring the input
split layout), plus a CSV of the removed rows for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ALIASES: dict[str, str] = {
    "arm": "ARM Assembly",
    "armassembly": "ARM Assembly",
    "armasm": "ARM Assembly",
    "aarch64": "ARM Assembly",
    "aarch64assembly": "ARM Assembly",
    "wolfram": "Mathematica/Wolfram Language",
    "wolframlanguage": "Mathematica/Wolfram Language",
    "mathematica": "Mathematica/Wolfram Language",
    "mathematicawolframlanguage": "Mathematica/Wolfram Language",
    "js": "JavaScript",
    "ts": "TypeScript",
    "objc": "Objective-C",
    "objectivec": "Objective-C",
    "cplusplus": "C++",
    "cpp": "C++",
    "csharp": "C#",
    "cs": "C#",
    "fsharp": "F#",
    "fs": "F#",
    "bash": "Shell",
    "sh": "Shell",
    "shellscript": "Shell",
    "batchscript": "Batchfile",
    "batch": "Batchfile",
    "emacslisp": "Emacs Lisp",
    "elisp": "Emacs Lisp",
    "commonlisp": "Common Lisp",
    "cl": "Common Lisp",
    "standardml": "Standard ML",
    "sml": "Standard ML",
    "vbnet": "Visual Basic .NET",
    "visualbasicnet": "Visual Basic .NET",
    "visualbasic": "Visual Basic .NET",
    "matlaboctave": "MATLAB",
    "octave": "MATLAB",
    "raku": "Raku",
    "perl6": "Raku",
    "nimrod": "Nim",
    "componentpascal": "Component Pascal",
    "modula2": "Modula-2",
    "modula3": "Modula-3",
    "powershellscript": "PowerShell",
    "ps1": "PowerShell",
    "rscript": "R",
}


def canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def resolve_predicted_to_canonical(
    predicted: str, key_to_canonical: dict[str, str]
) -> str | None:
    key = canonical_key(predicted)
    if key in key_to_canonical:
        return key_to_canonical[key]
    if key in ALIASES:
        alias = ALIASES[key]
        return key_to_canonical.get(canonical_key(alias))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-jsonl",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/validation/label_validation.jsonl",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(
            "~/llm_models/guardrail_code_data/processed/v1_splits"
        ).expanduser(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "~/llm_models/guardrail_code_data/processed/v1_splits_clean"
        ).expanduser(),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(
            "~/llm_models/guardrail_code_data/interim/reports/v1_languages.csv"
        ).expanduser(),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Match-rate threshold for 'mainstream' labels.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
        help="Optional path to write the Tier A dropped rows CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.audit_csv or args.output_dir / "tier_a_dropped.csv"

    labels_df = pd.read_csv(args.labels_csv)
    all_labels: set[str] = set(labels_df["canonical_language"].astype(str))
    key_to_canonical: dict[str, str] = {canonical_key(l): l for l in all_labels}

    per_label = defaultdict(lambda: [0, 0])  # [match, total]
    all_records: list[dict] = []
    with args.validation_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            lang = record.get("canonical_language")
            if not lang:
                continue
            per_label[lang][1] += 1
            if record.get("match"):
                per_label[lang][0] += 1
            all_records.append(record)

    mainstream: set[str] = {
        lang
        for lang, (m, t) in per_label.items()
        if t > 0 and (m / t) >= args.threshold
    }

    drop_ids: set[str] = set()
    audit_rows: list[dict] = []
    for record in all_records:
        if record.get("match") or record.get("error"):
            continue
        gold = record.get("canonical_language")
        if gold not in mainstream:
            continue
        predicted = record.get("predicted_language") or ""
        mapped = resolve_predicted_to_canonical(predicted, key_to_canonical)
        if mapped and mapped != gold and mapped in mainstream:
            drop_ids.add(record["id"])
            audit_rows.append(
                {
                    "id": record["id"],
                    "split": record.get("split", ""),
                    "source": record.get("source", ""),
                    "canonical_language": gold,
                    "predicted_language": predicted,
                    "mapped_to": mapped,
                }
            )

    print(f"mainstream labels (>= {args.threshold:.2f} match): {len(mainstream)}")
    print(f"Tier A rows to drop: {len(drop_ids)}")

    with audit_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "split", "source", "canonical_language", "predicted_language", "mapped_to"],
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"audit -> {audit_csv}")

    totals = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    for split in ("train", "val", "test"):
        in_path = args.splits_dir / f"{split}.parquet"
        out_path = args.output_dir / f"{split}.parquet"
        frame = pd.read_parquet(in_path)
        before = len(frame)
        filtered = frame[~frame["id"].isin(drop_ids)].copy()
        filtered.to_parquet(out_path, index=False)
        totals[split] = [before, len(filtered)]
        print(
            f"{split:5s}: {before} -> {len(filtered)} "
            f"(dropped {before - len(filtered)})  {out_path}"
        )

    summary = {
        "threshold": args.threshold,
        "mainstream_labels": sorted(mainstream),
        "mainstream_count": len(mainstream),
        "dropped_total": len(drop_ids),
        "splits": {
            split: {"before": b, "after": a, "dropped": b - a}
            for split, (b, a) in totals.items()
        },
    }
    with (args.output_dir / "filter_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"summary -> {args.output_dir / 'filter_summary.json'}")


if __name__ == "__main__":
    main()
