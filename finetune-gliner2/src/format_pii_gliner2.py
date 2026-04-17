#!/usr/bin/env python3
"""
Convert KDPII and synthetic_pii_finance_multilingual datasets into GLiNER2
training format for Korean PII NER.

GLiNER2 JSONL format:
{"input": "text", "output": {"entities": {"person": ["김민수"], "phone": ["010-1234-5678"]}}}

Usage:
    python3 format_pii_gliner2.py
    python3 format_pii_gliner2.py --max-per-dataset 10000
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

PROCESSED_DIR = Path("/data/processed") if Path("/data/processed").exists() else \
    Path(__file__).resolve().parents[2] / "korean-guardrail-dataset" / "data" / "processed"
OUTPUT_DIR = Path("/data") if Path("/data").exists() else \
    Path(__file__).resolve().parents[1] / "data"
SEED = 42

PII_DATASETS = ["KDPII", "synthetic_pii_finance_multilingual"]

# Extra synthetic files stored in OUTPUT_DIR (already in PII guardrail schema).
EXTRA_FILES = ["korean_rrn_synthetic.jsonl"]

# Consolidated to 10 core PII classes. Rare labels are dropped to improve
# per-class training signal. Low-value labels (date/time/location/country/age/
# measurement/grade/organization/job_title/gender/etc.) are skipped.
LABEL_MAP = {
    # KDPII Korean labels
    "QT_MOBILE": "phone",
    "QT_PHONE": "phone",
    "TMI_EMAIL": "email",
    "PS_NAME": "person",
    "PS_ID": "id_number",
    "LC_ADDRESS": "address",
    "DT_BIRTH": "date_of_birth",
    "QT_ACCOUNT_NUMBER": "bank_account",
    "QT_CARD_NUMBER": "credit_card",
    # gretelai labels (already English)
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

# Keep only these 10 high-value PII classes. Any label not in this set is dropped.
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

ENTITY_DESCRIPTIONS = {
    "person": "Full name of a person",
    "phone": "Phone or mobile number",
    "email": "Email address",
    "id_number": "Identity number, customer ID, or employee ID",
    "address": "Street address or mailing address",
    "date_of_birth": "Date of birth",
    "credit_card": "Credit or debit card number",
    "ssn": "Social security number or resident registration number",
    "passport": "Passport number",
    "driver_license": "Driver's license number",
    "bank_account": "Bank account number (IBAN/BBAN)",
}


def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label, label.lower())


# Filter by actual token count. Budget: 512 tokens total, reserve ~150 for
# GLiNER2 schema tokens (11 labels + descriptions), leaving ~360 for input.
MAX_INPUT_TOKENS = 360
TOKENIZER_NAME = "fastino/gliner2-multi-v1"

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def token_len(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def convert_record(row: dict) -> dict | None:
    """Convert a processed JSONL record to GLiNER2 NER format."""
    query = row.get("query", "").strip()
    if not query:
        return None

    # Drop records too long to fit in the encoder context after schema prepending.
    if token_len(query) > MAX_INPUT_TOKENS:
        return None

    raw_entities = row.get("answer", [])
    blocked = row.get("blocked", False)

    # Build entity dict: {label: [mention1, mention2, ...]}
    entities = {}
    descs_used = {}

    if isinstance(raw_entities, list):
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            form = ent.get("form", "").strip()
            label = ent.get("label", "")
            if not form or not label:
                continue

            # Verify span exists in text (GLiNER2 strict validation requires this)
            if form not in query:
                continue

            norm_label = normalize_label(label)
            # Drop any label not in the core PII set
            if norm_label not in ALLOWED_LABELS:
                continue
            if norm_label not in entities:
                entities[norm_label] = []
            # Avoid duplicate mentions
            if form not in entities[norm_label]:
                entities[norm_label].append(form)

            if norm_label in ENTITY_DESCRIPTIONS:
                descs_used[norm_label] = ENTITY_DESCRIPTIONS[norm_label]

    # For blocked records with no extracted entities, skip
    # (we can't train NER without entity annotations)
    if blocked and not entities:
        return None

    # For safe records (no PII), we still need at least one task
    # Use a classification to mark it as safe
    if not entities:
        return {
            "input": query,
            "output": {
                "entities": {},
                "classifications": [{
                    "task": "contains_pii",
                    "labels": ["yes", "no"],
                    "true_label": "no",
                }],
            }
        }

    result = {
        "input": query,
        "output": {
            "entities": entities,
        }
    }

    # Add descriptions for better zero-shot generalization
    if descs_used:
        result["output"]["entity_descriptions"] = descs_used

    # Also add PII classification task (multi-task training)
    result["output"]["classifications"] = [{
        "task": "contains_pii",
        "labels": ["yes", "no"],
        "true_label": "yes" if entities else "no",
    }]

    return result


def main():
    parser = argparse.ArgumentParser(description="Format PII datasets for GLiNER2 training")
    parser.add_argument("--max-per-dataset", type=int, default=0, help="Max records per dataset (0=all)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "pii_gliner2_train.jsonl"

    all_records = []
    stats = {}
    label_counts = Counter()

    dataset_sources = [(name, PROCESSED_DIR / f"{name}.jsonl") for name in PII_DATASETS]
    dataset_sources += [(Path(f).stem, OUTPUT_DIR / f) for f in EXTRA_FILES]

    for ds_name, ds_path in dataset_sources:
        if not ds_path.exists():
            print(f"  SKIP {ds_name} (not found at {ds_path})")
            continue

        records = []
        with ds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rec = convert_record(row)
                if rec:
                    records.append(rec)
                    for label, mentions in rec["output"].get("entities", {}).items():
                        label_counts[label] += len(mentions)

        # Sample if needed
        if args.max_per_dataset > 0 and len(records) > args.max_per_dataset:
            rng = random.Random(SEED)
            has_entities = [r for r in records if r["output"].get("entities")]
            no_entities = [r for r in records if not r["output"].get("entities")]
            ratio = len(has_entities) / len(records) if records else 0.5
            n_ent = round(args.max_per_dataset * ratio)
            n_safe = args.max_per_dataset - n_ent
            sampled = rng.sample(has_entities, min(n_ent, len(has_entities)))
            sampled += rng.sample(no_entities, min(n_safe, len(no_entities)))
            rng.shuffle(sampled)
            records = sampled

        n_with_ent = sum(1 for r in records if r["output"].get("entities"))
        stats[ds_name] = {"total": len(records), "with_entities": n_with_ent, "safe": len(records) - n_with_ent}
        all_records.extend(records)
        print(f"  {ds_name:<45} total={len(records):>6}  with_pii={n_with_ent:>6}  safe={len(records)-n_with_ent:>6}")

    # Shuffle
    rng = random.Random(SEED)
    rng.shuffle(all_records)

    # Write
    with output_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(all_records)
    print(f"\n{'='*70}")
    print(f"Total records:    {total}")
    print(f"Output:           {output_path}")
    print(f"\nEntity label distribution:")
    for label, count in label_counts.most_common(20):
        print(f"  {label:<25} {count:>6}")
    print(f"{'='*70}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({
            "per_dataset": stats,
            "total": total,
            "label_counts": dict(label_counts.most_common()),
        }, f, indent=2, ensure_ascii=False)
    print(f"Stats:            {stats_path}")


if __name__ == "__main__":
    main()
