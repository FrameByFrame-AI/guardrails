#!/usr/bin/env python3
"""
Benchmark GLiNER2 base model on Korean PII datasets (KDPII + synthetic_pii_finance)
before fine-tuning, to see if training is needed.

Metrics:
  - Binary: Does the model detect ANY PII in text? (blocked vs ground truth)
  - Entity-level: Precision/Recall/F1 on exact entity span matches

Usage (inside docker):
    python benchmark_pii.py
    python benchmark_pii.py --samples 500 --model fastino/gliner2-large-v1
"""

import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from gliner2 import GLiNER2

PII_LABELS = [
    "person", "phone number", "email address", "street address",
    "credit card number", "bank account number",
    "identity number", "social security number",
    "passport number", "driver license number",
    "ip address", "url", "username", "password",
    "date of birth", "organization",
]

# Map GLiNER2 predictions back to our normalized labels for comparison
PRED_LABEL_MAP = {
    "phone number": "phone",
    "email address": "email",
    "street address": "address",
    "credit card number": "credit_card",
    "bank account number": "bank_account",
    "identity number": "id_number",
    "social security number": "ssn",
    "passport number": "passport",
    "driver license number": "driver_license",
    "ip address": "ip_address",
    "date of birth": "date_of_birth",
}

# Map ground truth labels to normalized form
GT_LABEL_MAP = {
    "QT_MOBILE": "phone", "QT_PHONE": "phone",
    "TMI_EMAIL": "email", "TMI_SITE": "url",
    "PS_NAME": "person", "PS_NICKNAME": "username", "PS_ID": "id_number",
    "LC_ADDRESS": "address", "LC_PLACE": "location", "LCP_COUNTRY": "location",
    "DT_BIRTH": "date_of_birth",
    "QT_ACCOUNT_NUMBER": "bank_account", "QT_CARD_NUMBER": "credit_card",
    "QT_PLATE_NUMBER": "id_number",
    "OG_WORKPLACE": "organization", "OG_DEPARTMENT": "organization",
    "OGG_EDUCATION": "organization", "OGG_RELIGION": "organization",
    "OGG_CLUB": "organization",
    # gretelai
    "phone_number": "phone", "email": "email", "name": "person",
    "first_name": "person", "last_name": "person",
    "user_name": "username", "street_address": "address",
    "credit_card_number": "credit_card", "credit_card_security_code": "credit_card",
    "iban": "bank_account", "bban": "bank_account",
    "bank_routing_number": "bank_account", "swift_bic_code": "bank_account",
    "ssn": "ssn", "passport_number": "passport",
    "driver_license_number": "driver_license",
    "ipv4": "ip_address", "ipv6": "ip_address",
    "company": "organization", "customer_id": "id_number", "employee_id": "id_number",
    "date": "date", "date_of_birth": "date_of_birth", "date_time": "date",
    "local_latlng": "location", "password": "password", "api_key": "password",
    "account_pin": "password", "time": "date",
}


def norm_pred_label(label):
    return PRED_LABEL_MAP.get(label, label)


def norm_gt_label(label):
    return GT_LABEL_MAP.get(label, label.lower())


def load_dataset(path, max_samples, seed=42):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples > 0 and len(records) > max_samples:
        rng = random.Random(seed)
        blocked = [r for r in records if r.get("blocked")]
        safe = [r for r in records if not r.get("blocked")]
        ratio = len(blocked) / len(records)
        n_b = round(max_samples * ratio)
        n_s = max_samples - n_b
        sampled = rng.sample(blocked, min(n_b, len(blocked)))
        sampled += rng.sample(safe, min(n_s, len(safe)))
        rng.shuffle(sampled)
        return sampled
    return records


def benchmark_dataset(model, records, dataset_name):
    # Binary detection metrics
    tp = fp = tn = fn = 0

    # Entity-level metrics
    ent_tp = ent_fp = ent_fn = 0
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()

    t0 = time.time()

    for i, rec in enumerate(records):
        query = rec.get("query", "")
        gt_blocked = rec.get("blocked", False)
        gt_entities = rec.get("answer", [])

        # Get ground truth entity set: {(normalized_label, form)}
        gt_set = set()
        for ent in gt_entities:
            if isinstance(ent, dict) and "form" in ent and "label" in ent:
                gt_set.add((norm_gt_label(ent["label"]), ent["form"]))

        # Predict
        try:
            raw = model.extract_entities(query, PII_LABELS)
        except Exception as e:
            if i == 0:
                print(f"  extract_entities failed: {e}")
            raw = {}

        # Build prediction set — handle nested {"entities": {label: [mentions]}} format
        pred_set = set()
        ent_dict = raw
        if isinstance(raw, dict) and "entities" in raw:
            ent_dict = raw["entities"]

        if isinstance(ent_dict, dict):
            for label, mentions in ent_dict.items():
                if isinstance(mentions, list):
                    for m in mentions:
                        if isinstance(m, str) and m.strip():
                            pred_set.add((norm_pred_label(label), m.strip()))
        elif isinstance(raw, list):
            for p in raw:
                if isinstance(p, dict):
                    pred_set.add((norm_pred_label(p.get("label", "")), p.get("text", "")))

        pred_blocked = len(pred_set) > 0

        # Binary metrics
        if gt_blocked and pred_blocked:
            tp += 1
        elif gt_blocked and not pred_blocked:
            fn += 1
        elif not gt_blocked and pred_blocked:
            fp += 1
        else:
            tn += 1

        # Entity-level metrics (exact match on form text)
        gt_forms = {form for _, form in gt_set}
        pred_forms = {form for _, form in pred_set}

        matched = gt_forms & pred_forms
        ent_tp += len(matched)
        ent_fp += len(pred_forms - gt_forms)
        ent_fn += len(gt_forms - pred_forms)

        # Per-label breakdown
        for label, form in gt_set:
            if form in pred_forms:
                per_label_tp[label] += 1
            else:
                per_label_fn[label] += 1
        for label, form in pred_set:
            if form not in gt_forms:
                per_label_fp[label] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    [{dataset_name}] {i+1}/{len(records)} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    total = tp + fp + tn + fn

    # Binary metrics
    bin_acc = (tp + tn) / total if total else 0
    bin_prec = tp / (tp + fp) if (tp + fp) else 0
    bin_rec = tp / (tp + fn) if (tp + fn) else 0
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if (bin_prec + bin_rec) else 0

    # Entity-level metrics
    ent_prec = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) else 0
    ent_rec = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) else 0
    ent_f1 = 2 * ent_prec * ent_rec / (ent_prec + ent_rec) if (ent_prec + ent_rec) else 0

    return {
        "dataset": dataset_name,
        "total": total,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(elapsed / total * 1000, 1) if total else 0,
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
        "per_label": {
            label: {
                "tp": per_label_tp[label],
                "fp": per_label_fp[label],
                "fn": per_label_fn[label],
                "precision": round(per_label_tp[label] / (per_label_tp[label] + per_label_fp[label]), 4)
                    if (per_label_tp[label] + per_label_fp[label]) else 0,
                "recall": round(per_label_tp[label] / (per_label_tp[label] + per_label_fn[label]), 4)
                    if (per_label_tp[label] + per_label_fn[label]) else 0,
            }
            for label in sorted(set(list(per_label_tp.keys()) + list(per_label_fn.keys()) + list(per_label_fp.keys())))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--adapter", default="", help="Path to LoRA adapter directory")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--data-dir", default="/data/processed")
    parser.add_argument("--output", default="/data/benchmark_results/gliner2_pii_benchmark.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Prefer held-out test split if available, fall back to full files
    def pick(name):
        test = data_dir / f"{name}.test.jsonl"
        full = data_dir / f"{name}.jsonl"
        return test if test.exists() else full

    datasets = {
        "KDPII": pick("KDPII"),
        "synthetic_pii_finance": pick("synthetic_pii_finance_multilingual"),
    }

    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Samples per dataset: {args.samples}\n")
    print("Loading model...")
    model = GLiNER2.from_pretrained(args.model)
    if args.adapter:
        model.load_adapter(args.adapter)
        if hasattr(model, "merge_lora"):
            model.merge_lora()
        print("Adapter loaded and merged.")
    print("Model loaded.\n")

    all_results = {}

    for name, path in datasets.items():
        if not path.exists():
            print(f"  SKIP {name} (not found at {path})")
            continue

        records = load_dataset(str(path), args.samples)
        print(f"  [{name}] {len(records)} records")
        result = benchmark_dataset(model, records, name)
        all_results[name] = result

    # Print summary
    print(f"\n{'='*80}")
    print(f"GLiNER2 PII Benchmark Results — {args.model}")
    print(f"{'='*80}")
    print(f"\n{'DATASET':<30} {'BIN_ACC':>8} {'BIN_P':>8} {'BIN_R':>8} {'BIN_F1':>8} │ {'ENT_P':>8} {'ENT_R':>8} {'ENT_F1':>8} {'ms/rec':>8}")
    print("-" * 105)

    for name, r in all_results.items():
        b = r["binary"]
        e = r["entity_level"]
        print(f"{name:<30} {b['accuracy']:>8.4f} {b['precision']:>8.4f} {b['recall']:>8.4f} {b['f1']:>8.4f} │ {e['precision']:>8.4f} {e['recall']:>8.4f} {e['f1']:>8.4f} {r['ms_per_record']:>7.1f}")

    # Per-label breakdown for first dataset
    for name, r in all_results.items():
        print(f"\n  [{name}] Per-label breakdown:")
        print(f"  {'LABEL':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'PREC':>8} {'REC':>8}")
        print(f"  {'-'*65}")
        for label, m in sorted(r["per_label"].items(), key=lambda x: -(x[1]["tp"] + x[1]["fn"])):
            if m["tp"] + m["fn"] + m["fp"] == 0:
                continue
            print(f"  {label:<25} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['precision']:>8.4f} {m['recall']:>8.4f}")

    print(f"\n{'='*80}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
