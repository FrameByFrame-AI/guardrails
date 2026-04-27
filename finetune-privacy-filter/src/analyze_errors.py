#!/usr/bin/env python3
"""
Per-record error analysis for a fine-tuned Privacy Filter checkpoint.

For each gold span we ask: did the model produce an EXACT match, a BOUNDARY
mismatch (same label, overlapping but different start/end), a LABEL mismatch
(same boundaries, different category), an OVERLAP_AND_LABEL mismatch (overlap
with a different category), or did it MISS the span entirely?

For each predicted span we mirror the bucketing so we can quantify spurious
predictions versus boundary slips.

Output: per-label JSON breakdown plus a configurable number of human-readable
example error rows so we can eyeball the failure modes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--examples-per-bucket",
        type=int,
        default=10,
        help="Sample N concrete examples per error bucket per label.",
    )
    parser.add_argument(
        "--focus-labels",
        nargs="+",
        default=["private_person", "private_address"],
    )
    return parser.parse_args()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _normalize_id2label(mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in mapping.items()}


def _decode_bioes_spans(label_ids, id_to_label):
    spans = set()
    active_label = None
    active_start = None
    for token_idx, label_id in enumerate(label_ids):
        label_name = id_to_label[int(label_id)]
        if label_name == "O":
            if active_label is not None:
                spans.add((active_label, active_start, token_idx))
                active_label = None
                active_start = None
            continue
        prefix, category = label_name.split("-", 1)
        if prefix == "S":
            if active_label is not None:
                spans.add((active_label, active_start, token_idx))
            spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
        elif prefix == "B":
            if active_label is not None:
                spans.add((active_label, active_start, token_idx))
            active_label = category
            active_start = token_idx
        elif prefix == "I":
            if active_label != category or active_start is None:
                active_label = category
                active_start = token_idx
        elif prefix == "E":
            if active_label == category and active_start is not None:
                spans.add((category, active_start, token_idx + 1))
            else:
                spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
    if active_label is not None:
        spans.add((active_label, active_start, len(label_ids)))
    return spans


def _predict(text, tokenizer, model, id_to_label, max_length):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    if torch.cuda.is_available():
        enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    pred_ids = out.logits.float().detach().cpu().numpy().argmax(axis=-1)[0].tolist()
    token_spans = _decode_bioes_spans(pred_ids, id_to_label)
    char_spans = []
    for label, ts, te in token_spans:
        if te <= ts:
            continue
        char_start = int(offsets[ts][0])
        char_end = int(offsets[te - 1][1])
        if char_end <= char_start:
            continue
        char_spans.append((label, char_start, char_end))
    return char_spans


def _gold_spans(record):
    out = []
    label_items = record.get("label")
    if isinstance(label_items, list):
        for item in label_items:
            if isinstance(item, dict):
                out.append((str(item["category"]), int(item["start"]), int(item["end"])))
    return out


def _overlap(a, b):
    return not (a[2] <= b[1] or b[2] <= a[1])


def classify_gold(gold, preds):
    g_label, g_start, g_end = gold
    overlapping = [p for p in preds if _overlap(gold, p)]
    if not overlapping:
        return "MISSED", None
    exact = [p for p in overlapping if p == gold]
    if exact:
        return "EXACT", exact[0]
    same_label_overlap = [p for p in overlapping if p[0] == g_label]
    if same_label_overlap:
        same_boundary = [p for p in same_label_overlap if p[1] == g_start and p[2] == g_end]
        if same_boundary:
            return "EXACT", same_boundary[0]
        return "BOUNDARY", same_label_overlap[0]
    same_boundary_diff_label = [p for p in overlapping if p[1] == g_start and p[2] == g_end]
    if same_boundary_diff_label:
        return "WRONG_LABEL", same_boundary_diff_label[0]
    return "OVERLAP_AND_LABEL", overlapping[0]


def classify_pred(pred, golds):
    p_label, p_start, p_end = pred
    overlapping = [g for g in golds if _overlap(pred, g)]
    if not overlapping:
        return "SPURIOUS", None
    exact = [g for g in overlapping if g == pred]
    if exact:
        return "EXACT", exact[0]
    same_label_overlap = [g for g in overlapping if g[0] == p_label]
    if same_label_overlap:
        return "BOUNDARY", same_label_overlap[0]
    same_boundary_diff_label = [g for g in overlapping if g[1] == p_start and g[2] == p_end]
    if same_boundary_diff_label:
        return "WRONG_LABEL", same_boundary_diff_label[0]
    return "OVERLAP_AND_LABEL", overlapping[0]


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, torch_dtype="auto", trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    id_to_label = _normalize_id2label(model.config.id2label)
    records = _load_jsonl(args.dataset)

    gold_buckets = defaultdict(Counter)
    pred_buckets = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))

    for record in records:
        text = str(record.get("text", ""))
        golds = _gold_spans(record)
        preds = _predict(text, tokenizer, model, id_to_label, args.max_length)

        for g in golds:
            bucket, matched = classify_gold(g, preds)
            gold_buckets[g[0]][bucket] += 1
            if (
                g[0] in args.focus_labels
                and len(examples[g[0]][f"gold_{bucket}"]) < args.examples_per_bucket
            ):
                examples[g[0]][f"gold_{bucket}"].append(
                    {
                        "text": text,
                        "gold": {"label": g[0], "start": g[1], "end": g[2], "form": text[g[1]:g[2]]},
                        "matched_pred": (
                            None
                            if matched is None
                            else {
                                "label": matched[0],
                                "start": matched[1],
                                "end": matched[2],
                                "form": text[matched[1]:matched[2]],
                            }
                        ),
                    }
                )

        for p in preds:
            bucket, matched = classify_pred(p, golds)
            pred_buckets[p[0]][bucket] += 1
            if (
                p[0] in args.focus_labels
                and bucket != "EXACT"
                and len(examples[p[0]][f"pred_{bucket}"]) < args.examples_per_bucket
            ):
                examples[p[0]][f"pred_{bucket}"].append(
                    {
                        "text": text,
                        "pred": {"label": p[0], "start": p[1], "end": p[2], "form": text[p[1]:p[2]]},
                        "matched_gold": (
                            None
                            if matched is None
                            else {
                                "label": matched[0],
                                "start": matched[1],
                                "end": matched[2],
                                "form": text[matched[1]:matched[2]],
                            }
                        ),
                    }
                )

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "gold_breakdown": {
            label: dict(buckets) for label, buckets in sorted(gold_buckets.items())
        },
        "pred_breakdown": {
            label: dict(buckets) for label, buckets in sorted(pred_buckets.items())
        },
        "examples": {label: dict(buckets) for label, buckets in examples.items()},
    }

    print("=== Gold span breakdown (where each gold span ended up) ===")
    for label, buckets in summary["gold_breakdown"].items():
        total = sum(buckets.values())
        cells = " ".join(f"{k}={v}" for k, v in sorted(buckets.items()))
        print(f"  {label:<18} total={total:<5} {cells}")
    print("\n=== Pred span breakdown (what each prediction overlapped with) ===")
    for label, buckets in summary["pred_breakdown"].items():
        total = sum(buckets.values())
        cells = " ".join(f"{k}={v}" for k, v in sorted(buckets.items()))
        print(f"  {label:<18} total={total:<5} {cells}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
