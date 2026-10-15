#!/usr/bin/env python3
"""
Convert a hand-authored PII redaction eval log (CSV: Input_Text +
Expected_Output with [LABEL] redaction placeholders) into OPF JSONL, for use
as an independent, out-of-KDPII-distribution eval set alongside
ko_pii_opf_v4/test.jsonl.

Character-offset spans aren't given directly -- they're recovered by
aligning Input_Text against Expected_Output: splitting Expected_Output on
`[LABEL]` placeholders gives an alternating list of literal segments (which
must appear verbatim and in order in Input_Text) and tags; the gap between
consecutive literal segments in Input_Text is the entity span for the tag
between them.

This alignment fails on inputs where redaction also rewrote grammar around
the placeholder (e.g. Korean particle allomorphy: 을/를, 이/가, 은/는, 과/와
changing to agree with the replacement rather than the original word) --
such rows are skipped and counted, not silently dropped from the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


LABEL_MAP = {
    "PRIVATE_PERSON": "private_person",
    "PERSONAL_HANDLE": "personal_handle",
    "PRIVATE_PHONE": "private_phone",
    "PRIVATE_EMAIL": "private_email",
    "PRIVATE_ADDRESS": "private_address",
    "PRIVATE_DATE": "private_date",
    "PRIVATE_URL": "private_url",
    "ACCOUNT_NUMBER": "account_number",
    "PRIVATE_ACCOUNT": "account_number",
    "PRIVATE_ID": "account_number",
    "IP_ADDRESS": "ip_address",
    "PRIVATE_IP": "ip_address",
    "LOCATION_COORDINATES": "private_address",
    # No correspondence in this project's 9-category schema; dropped.
    "PRIVATE_ORGANIZATION": None,
}

# The CSV mixes UPPER_SNAKE and lower_snake placeholder casing inconsistently
# (some rows already use this project's own category names verbatim); matched
# case-insensitively and normalized to upper before the LABEL_MAP lookup.
PLACEHOLDER_RE = re.compile(r"\[([A-Za-z_]+)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting a non-empty --output-dir.")
    return parser.parse_args()


def align_spans(text: str, expected: str) -> tuple[list[tuple[str, int, int]], bool]:
    """Recover (raw_label, start, end) spans in `text` from `expected`'s
    [LABEL] placeholders. Returns (spans, ok) -- ok=False means alignment
    broke down (a literal segment from `expected` wasn't found in order in
    `text`) and the row should be skipped."""
    parts = PLACEHOLDER_RE.split(expected)
    spans: list[tuple[str, int, int]] = []
    cursor = 0

    if not text.startswith(parts[0]):
        return [], False
    cursor += len(parts[0])

    for i in range(1, len(parts), 2):
        tag = parts[i]
        literal_after = parts[i + 1]
        end = len(text) if not literal_after else text.find(literal_after, cursor)
        if end == -1:
            return [], False
        spans.append((tag, cursor, end))
        cursor = end + len(literal_after)

    return spans, True


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"--output-dir {args.output_dir} already exists and is non-empty. "
            "Pass --force to overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict[str, Any]] = []
    alignment_failures: list[str] = []
    dropped_unmapped_labels: dict[str, int] = {}

    for row in rows:
        test_id = row.get("Test_Case_ID", "")
        text = row["Input_Text"]
        expected = row["Expected_Output"]

        spans, ok = align_spans(text, expected)
        if not ok:
            alignment_failures.append(test_id)
            continue

        labels: list[dict[str, Any]] = []
        for raw_label, start, end in spans:
            normalized_label = raw_label.upper()
            if normalized_label not in LABEL_MAP:
                raise ValueError(f"{test_id}: unrecognized placeholder [{raw_label}]")
            target_label = LABEL_MAP[normalized_label]
            if target_label is None:
                dropped_unmapped_labels[raw_label] = dropped_unmapped_labels.get(raw_label, 0) + 1
                continue
            if text[start:end] != text[start:end].strip() or start == end:
                # Defensive: alignment produced a degenerate/whitespace-only span.
                continue
            labels.append({"category": target_label, "start": start, "end": end})

        labels.sort(key=lambda item: (item["start"], item["end"], item["category"]))
        records.append({
            "text": text,
            "label": labels,
            "info": {"source_dataset": "pii_model_evaluation_log", "source_id": test_id},
        })

    output_path = args.output_dir / "test.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    label_space_path = args.output_dir / "label_space.json"
    label_space_path.write_text(
        json.dumps(
            {
                "category_version": "pii_eval_log_v1",
                "span_class_names": ["O"] + sorted(set(LABEL_MAP.values()) - {None}),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_csv": str(args.input_csv),
        "total_rows": len(rows),
        "converted_records": len(records),
        "alignment_failures": alignment_failures,
        "dropped_unmapped_labels": dropped_unmapped_labels,
        "total_gold_spans": sum(len(r["label"]) for r in records),
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
