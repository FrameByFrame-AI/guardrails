#!/usr/bin/env python3
"""
Benchmark GLiNER2 fine-tuned model on the held-out test split.

Aligned with training:
  - Same 11 ALLOWED_LABELS that we trained on (drop others from both pred + GT)
  - Same LABEL_MAP normalization as format_pii_gliner2.py
  - Reads *.test.jsonl from the split folder

Usage:
    python benchmark_pii_heldout.py \
        --model fastino/gliner2-multi-v1 \
        --adapter /models/gliner2-multi-korean-pii-lora/checkpoint-epoch-6 \
        --data-dir /data/processed \
        --output /data/benchmark_results/gliner2_pii_heldout_benchmark.json
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from gliner2 import GLiNER2


# Exactly the 11 classes used in training (format_pii_gliner2.py ALLOWED_LABELS)
ALLOWED_LABELS = {
    "person",
    "phone",
    "email",
    "address",
    "date_of_birth",
    "credit_card",
    "bank_account",
    "ssn",
    "passport",
    "driver_license",
    "id_number",
}

# What we prompt GLiNER2 with (natural-language labels) and how they map back
PROMPT_LABELS = [
    "person",
    "phone number",
    "email address",
    "street address",
    "date of birth",
    "credit card number",
    "bank account number",
    "social security number",
    "passport number",
    "driver license number",
    "identity number",
]

PROMPT_TO_NORM = {
    "person": "person",
    "phone number": "phone",
    "email address": "email",
    "street address": "address",
    "date of birth": "date_of_birth",
    "credit card number": "credit_card",
    "bank account number": "bank_account",
    "social security number": "ssn",
    "passport number": "passport",
    "driver license number": "driver_license",
    "identity number": "id_number",
}

# Same mapping as training (format_pii_gliner2.py LABEL_MAP) — ground truth normalization.
GT_LABEL_MAP = {
    # KDPII Korean labels (uppercase)
    "QT_MOBILE": "phone",
    "QT_PHONE": "phone",
    "TMI_EMAIL": "email",
    "PS_NAME": "person",
    "PS_ID": "id_number",
    "LC_ADDRESS": "address",
    "DT_BIRTH": "date_of_birth",
    "QT_ACCOUNT_NUMBER": "bank_account",
    "QT_CARD_NUMBER": "credit_card",
    # gretelai labels (lowercase)
    "phone_number": "phone",
    "email": "email",
    "name": "person",
    "first_name": "person",
    "last_name": "person",
    "street_address": "address",
    "date_of_birth": "date_of_birth",
    "ssn": "ssn",
    "passport_number": "passport",
    "driver_license_number": "driver_license",
    "credit_card_number": "credit_card",
    "iban": "bank_account",
    "bban": "bank_account",
    "customer_id": "id_number",
    "employee_id": "id_number",
}


def normalize_gt_label(label: str) -> str | None:
    """Return the normalized label or None if it's a class we didn't train on."""
    normalized = GT_LABEL_MAP.get(label)
    if normalized and normalized in ALLOWED_LABELS:
        return normalized
    return None


def normalize_pred_label(label: str) -> str | None:
    """Map model's predicted label to our canonical form."""
    normalized = PROMPT_TO_NORM.get(label)
    if normalized and normalized in ALLOWED_LABELS:
        return normalized
    return None


def load_dataset(path: Path, max_samples: int, seed: int = 42):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples > 0 and len(records) > max_samples:
        rng = random.Random(seed)
        blocked = [r for r in records if r.get("blocked")]
        safe = [r for r in records if not r.get("blocked")]
        ratio = len(blocked) / len(records) if records else 0.5
        n_b = round(max_samples * ratio)
        n_s = max_samples - n_b
        sampled = rng.sample(blocked, min(n_b, len(blocked)))
        sampled += rng.sample(safe, min(n_s, len(safe)))
        rng.shuffle(sampled)
        return sampled
    return records


def has_trained_pii(record) -> bool:
    """Record contains at least one entity in ALLOWED_LABELS (after normalization)."""
    for ent in record.get("answer", []) or []:
        if isinstance(ent, dict) and normalize_gt_label(ent.get("label", "")):
            return True
    return False


def benchmark_dataset(model, records, dataset_name: str):
    # Binary detection (is there ANY trained-class PII?)
    tp = fp = tn = fn = 0

    # Entity-level (matched on (label, form) after normalization)
    ent_tp = ent_fp = ent_fn = 0
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()

    t0 = time.time()

    for i, rec in enumerate(records):
        query = rec.get("query", "")

        # Ground truth — only count entities from classes we trained on
        gt_set = set()
        for ent in rec.get("answer", []) or []:
            if not isinstance(ent, dict):
                continue
            form = ent.get("form", "").strip()
            norm = normalize_gt_label(ent.get("label", ""))
            if form and norm:
                gt_set.add((norm, form))
        gt_blocked = len(gt_set) > 0

        # Predict
        try:
            raw = model.extract_entities(query, PROMPT_LABELS)
        except Exception as e:
            if i == 0:
                print(f"  extract_entities failed: {e}")
            raw = {}

        # Normalize prediction into {(norm_label, form)}
        pred_set = set()
        ent_dict = raw.get("entities", raw) if isinstance(raw, dict) else raw
        if isinstance(ent_dict, dict):
            for label, mentions in ent_dict.items():
                norm = normalize_pred_label(label)
                if not norm:
                    continue
                if isinstance(mentions, list):
                    for m in mentions:
                        if isinstance(m, str) and m.strip():
                            pred_set.add((norm, m.strip()))
        elif isinstance(raw, list):
            for p in raw:
                if isinstance(p, dict):
                    norm = normalize_pred_label(p.get("label", ""))
                    form = p.get("text", "").strip()
                    if norm and form:
                        pred_set.add((norm, form))

        pred_blocked = len(pred_set) > 0

        # Binary
        if gt_blocked and pred_blocked:
            tp += 1
        elif gt_blocked and not pred_blocked:
            fn += 1
        elif not gt_blocked and pred_blocked:
            fp += 1
        else:
            tn += 1

        # Entity-level (label + form must both match)
        matched = gt_set & pred_set
        ent_tp += len(matched)
        ent_fp += len(pred_set - gt_set)
        ent_fn += len(gt_set - pred_set)

        # Per-label breakdown
        for label, form in gt_set:
            if (label, form) in pred_set:
                per_label_tp[label] += 1
            else:
                per_label_fn[label] += 1
        for label, form in pred_set:
            if (label, form) not in gt_set:
                per_label_fp[label] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{dataset_name}] {i+1}/{len(records)} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    total = tp + fp + tn + fn

    bin_prec = tp / (tp + fp) if (tp + fp) else 0.0
    bin_rec = tp / (tp + fn) if (tp + fn) else 0.0
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) else 0.0
    bin_acc = (tp + tn) / total if total else 0.0

    ent_prec = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) else 0.0
    ent_rec = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) else 0.0
    ent_f1 = 2 * ent_prec * ent_rec / (ent_prec + ent_rec) if (ent_prec + ent_rec) else 0.0

    per_label = {}
    for label in sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn)):
        lt = per_label_tp[label]
        lp = per_label_fp[label]
        lf = per_label_fn[label]
        p = lt / (lt + lp) if (lt + lp) else 0.0
        r = lt / (lt + lf) if (lt + lf) else 0.0
        per_label[label] = {"tp": lt, "fp": lp, "fn": lf,
                            "precision": round(p, 4), "recall": round(r, 4)}

    return {
        "dataset": dataset_name,
        "total": total,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / total, 1) if total else 0,
        "binary": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(bin_acc, 4),
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1": round(bin_f1, 4),
        },
        "entity_level": {
            "tp": ent_tp, "fp": ent_fp, "fn": ent_fn,
            "precision": round(ent_prec, 4),
            "recall": round(ent_rec, 4),
            "f1": round(ent_f1, 4),
        },
        "per_label": per_label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--adapter", default="",
                        help="Path to LoRA adapter (e.g. /models/<run>/checkpoint-epoch-6)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--data-dir", default="/data/processed")
    parser.add_argument("--output", default="/data/benchmark_results/gliner2_pii_heldout_benchmark.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    def pick(name):
        test = data_dir / f"{name}.test.jsonl"
        full = data_dir / f"{name}.jsonl"
        if test.exists():
            return test
        print(f"WARNING: {test} not found, falling back to {full} (possible train/test leak)")
        return full

    datasets = {
        "KDPII": pick("KDPII"),
        "synthetic_pii_finance": pick("synthetic_pii_finance_multilingual"),
    }

    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Allowed labels ({len(ALLOWED_LABELS)}): {sorted(ALLOWED_LABELS)}")
    print(f"Samples per dataset: {args.samples}\n")

    print("Loading model...")
    model = GLiNER2.from_pretrained(args.model)
    if args.adapter:
        model.load_adapter(args.adapter)
        if hasattr(model, "merge_lora"):
            model.merge_lora()
    print(f"Loaded.\n")

    results = {}
    for name, path in datasets.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        records = load_dataset(path, args.samples)
        if not records:
            continue

        # Drop records where all entities are classes we didn't train on —
        # these can't be fairly judged for trained-class recall.
        original_count = len(records)
        records = [r for r in records
                   if not r.get("blocked") or has_trained_pii(r)]
        if len(records) < original_count:
            print(f"  [{name}] Kept {len(records)}/{original_count} records "
                  f"(dropped {original_count - len(records)} with only non-trained classes)")

        print(f"Benchmarking {name} ({len(records)} records)...", flush=True)
        results[name] = benchmark_dataset(model, records, name)
        print()

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"{'DATASET':<30} {'BIN F1':>8} {'BIN P':>8} {'BIN R':>8}  "
          f"{'ENT F1':>8} {'ENT P':>8} {'ENT R':>8}")
    print("-" * 80)
    for name, r in results.items():
        b, e = r["binary"], r["entity_level"]
        print(f"{name:<30} {b['f1']:>8.4f} {b['precision']:>8.4f} {b['recall']:>8.4f}  "
              f"{e['f1']:>8.4f} {e['precision']:>8.4f} {e['recall']:>8.4f}")
    print("=" * 80)

    # Per-label table
    for name, r in results.items():
        print(f"\n  [{name}] Per-label breakdown:")
        print(f"  {'LABEL':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'PREC':>8} {'REC':>8}")
        print(f"  {'-' * 60}")
        for label, stats in sorted(r["per_label"].items(),
                                   key=lambda x: -(x[1]['tp'] + x[1]['fn'])):
            print(f"  {label:<20} {stats['tp']:>6} {stats['fp']:>6} {stats['fn']:>6} "
                  f"{stats['precision']:>8.4f} {stats['recall']:>8.4f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
