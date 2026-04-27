#!/usr/bin/env python3
"""
Convert Korean PII datasets into OpenAI Privacy Filter finetuning JSONL.

The converter currently supports:
- KDPII
- korean_rrn_synthetic

Output records use the OPF `label` schema:
{
  "text": "...",
  "label": [{"category": "...", "start": 0, "end": 3}],
  "info": {...}
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_SPAN_CLASS_NAMES = (
    "O",
    "private_person",
    "personal_handle",
    "private_phone",
    "private_email",
    "private_address",
    "private_date",
    "private_url",
    "account_number",
    "ip_address",
)

SOURCE_LABEL_MAP = {
    # KDPII
    "PS_NAME": "private_person",
    "PS_NICKNAME": "private_person",
    "PS_ID": "personal_handle",
    "QT_PHONE": "private_phone",
    "QT_MOBILE": "private_phone",
    "TMI_EMAIL": "private_email",
    "LC_ADDRESS": "private_address",
    "DT_BIRTH": "private_date",
    "TMI_SITE": "private_url",
    "QT_ACCOUNT_NUMBER": "account_number",
    "QT_CARD_NUMBER": "account_number",
    "QT_RESIDENT_NUMBER": "account_number",
    "QT_PASSPORT_NUMBER": "account_number",
    "QT_DRIVER_NUMBER": "account_number",
    "QT_ALIEN_NUMBER": "account_number",
    "QT_PLATE_NUMBER": "account_number",
    "QT_IP": "ip_address",
    # korean_rrn_synthetic
    "person": "private_person",
    "ssn": "account_number",
}

AMBIGUOUS_ADDRESS_TOPONYMS = {
    "한국",
    "서울",
    "부산",
    "수원",
}


@dataclass(frozen=True)
class SourceAnnotation:
    source_label: str
    target_label: str
    form: str
    source_index: int


@dataclass(frozen=True)
class ConversionError:
    reason: str
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Korean PII datasets into OPF finetuning JSONL."
    )
    parser.add_argument(
        "--kdpii",
        type=Path,
        default=Path(
            "/home/vijay/workspace/projects/accuknox/guardrail/finetune-qwen/processed/KDPII.jsonl"
        ),
        help="Path to KDPII.jsonl",
    )
    parser.add_argument(
        "--rrn",
        type=Path,
        default=Path(
            "/home/vijay/workspace/projects/accuknox/guardrail/finetune-gliner2/data/korean_rrn_synthetic.jsonl"
        ),
        help="Path to korean_rrn_synthetic.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/vijay/workspace/projects/accuknox/guardrail/finetune-privacy-filter/data/generated/ko_pii_opf_v4"
        ),
        help="Output directory for converted OPF files",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio for KDPII records",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Test split ratio for KDPII records",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic text-hash split seed for KDPII records",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def find_all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index < 0:
            break
        matches.append((index, index + len(needle)))
        cursor = index + 1
    return matches


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def stable_bucket(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(1 << 64)


def choose_kdpii_split(text: str, validation_ratio: float, test_ratio: float, seed: int) -> str:
    bucket = stable_bucket(text, seed)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + validation_ratio:
        return "validation"
    return "train"


def build_label_space_payload() -> dict[str, Any]:
    return {
        "category_version": "ko_pii_v4",
        "span_class_names": list(TARGET_SPAN_CLASS_NAMES),
    }


def build_source_annotations(record: dict[str, Any]) -> tuple[list[SourceAnnotation], Counter[str], bool]:
    raw_answers = record.get("answer") or []
    if not isinstance(raw_answers, list):
        raise ValueError("record.answer must be a list when present")

    mapped: list[SourceAnnotation] = []
    dropped_labels: Counter[str] = Counter()
    had_any_source_annotations = False

    for index, raw_annotation in enumerate(raw_answers):
        if not isinstance(raw_annotation, dict):
            raise ValueError(f"record.answer[{index}] must be an object")
        form = raw_annotation.get("form")
        label = raw_annotation.get("label")
        if not isinstance(form, str) or not isinstance(label, str):
            raise ValueError(f"record.answer[{index}] must contain string form/label")
        had_any_source_annotations = True
        if label == "PS_NAME" and len(form) == 1:
            dropped_labels["PS_NAME::single_char"] += 1
            continue
        if label == "LC_ADDRESS" and form in AMBIGUOUS_ADDRESS_TOPONYMS:
            dropped_labels["LC_ADDRESS::ambiguous_toponym"] += 1
            continue

        target_label = SOURCE_LABEL_MAP.get(label)
        if target_label is None:
            dropped_labels[label] += 1
            continue
        mapped.append(
            SourceAnnotation(
                source_label=label,
                target_label=target_label,
                form=form,
                source_index=index,
            )
        )
    return mapped, dropped_labels, had_any_source_annotations


def assign_char_spans(
    text: str,
    annotations: list[SourceAnnotation],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    occurrence_cache = {ann.form: find_all_occurrences(text, ann.form) for ann in annotations}
    used_spans: list[tuple[int, int]] = []
    assigned: list[dict[str, Any]] = []
    stats = Counter()

    for annotation in sorted(annotations, key=lambda ann: (-len(ann.form), ann.source_index)):
        occurrences = occurrence_cache[annotation.form]
        if not occurrences:
            raise ValueError(
                f"Exact text match not found for form={annotation.form!r} label={annotation.source_label!r}"
            )

        available = [
            occurrence
            for occurrence in occurrences
            if not any(spans_overlap(occurrence, used_span) for used_span in used_spans)
        ]
        if not available:
            raise ValueError(
                f"No non-overlapping match available for form={annotation.form!r} "
                f"label={annotation.source_label!r}"
            )
        if len(occurrences) > 1:
            stats["multi_match_forms"] += 1
        if len(available) > 1:
            stats["ambiguous_assignments_resolved_left_to_right"] += 1

        start, end = available[0]
        used_spans.append((start, end))
        assigned.append(
            {
                "category": annotation.target_label,
                "start": start,
                "end": end,
                "_source_form": annotation.form,
                "_source_label": annotation.source_label,
            }
        )

    assigned.sort(key=lambda item: (item["start"], item["end"], item["category"]))
    for item in assigned:
        if text[item["start"] : item["end"]] != item["_source_form"]:
            raise ValueError(
                "Assigned span text mismatch for "
                f"form={item['_source_form']!r} category={item['category']!r}"
            )
    cleaned = [
        {"category": item["category"], "start": item["start"], "end": item["end"]}
        for item in assigned
    ]
    return cleaned, stats


def postprocess_spans(
    text: str, labels: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    trimmed: list[dict[str, Any]] = []
    for item in labels:
        start, end = int(item["start"]), int(item["end"])
        while start < end and text[start].isspace():
            start += 1
            stats["postprocess::stripped_leading_ws"] += 1
        while end > start and text[end - 1].isspace():
            end -= 1
            stats["postprocess::stripped_trailing_ws"] += 1
        if start >= end:
            stats["postprocess::dropped_empty_after_strip"] += 1
            continue
        trimmed.append({"category": item["category"], "start": start, "end": end})

    trimmed.sort(key=lambda it: (it["start"], it["end"], it["category"]))
    return trimmed, stats


def convert_record(
    *,
    dataset_name: str,
    record: dict[str, Any],
    line_no: int,
) -> tuple[dict[str, Any] | None, ConversionError | None, Counter[str], Counter[str]]:
    stats = Counter()
    dropped_label_counts = Counter()

    text = record.get("query")
    if not isinstance(text, str) or not text:
        return (
            None,
            ConversionError(
                reason="invalid_text",
                details={"dataset": dataset_name, "line_no": line_no},
            ),
            stats,
            dropped_label_counts,
        )

    try:
        mapped_annotations, dropped_label_counts, had_any_source_annotations = build_source_annotations(record)
    except ValueError as exc:
        return (
            None,
            ConversionError(
                reason="invalid_annotation_schema",
                details={
                    "dataset": dataset_name,
                    "line_no": line_no,
                    "error": str(exc),
                },
            ),
            stats,
            dropped_label_counts,
        )

    if had_any_source_annotations and not mapped_annotations:
        return (
            None,
            ConversionError(
                reason="unsupported_only_annotations",
                details={
                    "dataset": dataset_name,
                    "line_no": line_no,
                    "record_id": record.get("id"),
                    "dropped_labels": dict(dropped_label_counts),
                },
            ),
            stats,
            dropped_label_counts,
        )

    if mapped_annotations:
        try:
            labels, assignment_stats = assign_char_spans(text, mapped_annotations)
        except ValueError as exc:
            return (
                None,
                ConversionError(
                    reason="span_assignment_failed",
                    details={
                        "dataset": dataset_name,
                        "line_no": line_no,
                        "record_id": record.get("id"),
                        "error": str(exc),
                    },
                ),
                stats,
                dropped_label_counts,
            )
        stats.update(assignment_stats)
    else:
        labels = []

    labels, postprocess_stats = postprocess_spans(text, labels)
    stats.update(postprocess_stats)

    for label in labels:
        stats[f"target_label::{label['category']}"] += 1
    for annotation in mapped_annotations:
        stats[f"source_label::{annotation.source_label}"] += 1

    converted = {
        "text": text,
        "label": labels,
        "info": {
            "source_dataset": dataset_name,
            "source_id": record.get("id"),
            "source_line_no": line_no,
            "source_blocked": bool(record.get("blocked", False)),
        },
    }
    return converted, None, stats, dropped_label_counts


def write_jsonl_record(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.validation_ratio < 0 or args.test_ratio < 0:
        raise SystemExit("validation/test ratios must be non-negative")
    if args.validation_ratio + args.test_ratio >= 1.0:
        raise SystemExit("validation_ratio + test_ratio must be < 1.0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    test_path = args.output_dir / "test.jsonl"
    all_path = args.output_dir / "all.jsonl"
    rejections_path = args.output_dir / "conversion_rejections.jsonl"
    summary_path = args.output_dir / "conversion_summary.json"
    label_space_path = args.output_dir / "label_space.json"

    stats = Counter()
    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    rejection_reasons = Counter()
    dropped_source_labels = Counter()

    with (
        train_path.open("w", encoding="utf-8") as train_handle,
        validation_path.open("w", encoding="utf-8") as validation_handle,
        test_path.open("w", encoding="utf-8") as test_handle,
        all_path.open("w", encoding="utf-8") as all_handle,
        rejections_path.open("w", encoding="utf-8") as rejection_handle,
    ):
        split_handles = {
            "train": train_handle,
            "validation": validation_handle,
            "test": test_handle,
        }

        dataset_sources: list[tuple[str, Path]] = [
            ("KDPII", args.kdpii),
            ("korean_rrn_synthetic", args.rrn),
        ]

        for dataset_name, path in dataset_sources:
            for line_no, record in iter_jsonl(path):
                stats[f"records_seen::{dataset_name}"] += 1
                converted, error, record_stats, record_dropped_labels = convert_record(
                    dataset_name=dataset_name,
                    record=record,
                    line_no=line_no,
                )
                stats.update(record_stats)
                dropped_source_labels.update(record_dropped_labels)
                if error is not None:
                    rejection_reasons[error.reason] += 1
                    write_jsonl_record(
                        rejection_handle,
                        {
                            "reason": error.reason,
                            **error.details,
                        },
                    )
                    continue

                if dataset_name == "KDPII":
                    split_name = choose_kdpii_split(
                        converted["text"],
                        validation_ratio=args.validation_ratio,
                        test_ratio=args.test_ratio,
                        seed=args.split_seed,
                    )
                else:
                    split_name = "train"

                split_counts[split_name][dataset_name] += 1
                write_jsonl_record(split_handles[split_name], converted)
                write_jsonl_record(all_handle, converted)

    label_space_path.write_text(
        json.dumps(build_label_space_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "inputs": {
            "kdpii": str(args.kdpii),
            "rrn": str(args.rrn),
        },
        "output_dir": str(args.output_dir),
        "split_seed": args.split_seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "target_span_class_names": list(TARGET_SPAN_CLASS_NAMES),
        "source_to_target_label_map": SOURCE_LABEL_MAP,
        "records_seen": {
            "KDPII": stats["records_seen::KDPII"],
            "korean_rrn_synthetic": stats["records_seen::korean_rrn_synthetic"],
        },
        "split_counts": {
            split_name: dict(counter)
            for split_name, counter in split_counts.items()
        },
        "target_label_counts": {
            key.split("::", 1)[1]: value
            for key, value in sorted(stats.items())
            if key.startswith("target_label::")
        },
        "source_label_counts_used": {
            key.split("::", 1)[1]: value
            for key, value in sorted(stats.items())
            if key.startswith("source_label::")
        },
        "dropped_source_label_counts": dict(sorted(dropped_source_labels.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "assignment_stats": {
            "multi_match_forms": stats["multi_match_forms"],
            "ambiguous_assignments_resolved_left_to_right": stats[
                "ambiguous_assignments_resolved_left_to_right"
            ],
        },
        "postprocess_stats": {
            key.split("::", 1)[1]: value
            for key, value in sorted(stats.items())
            if key.startswith("postprocess::")
        },
        "artifacts": {
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "all": str(all_path),
            "label_space": str(label_space_path),
            "rejections": str(rejections_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
