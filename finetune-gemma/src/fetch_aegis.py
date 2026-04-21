#!/usr/bin/env python3
"""
Download and convert NVIDIA Aegis-AI-Content-Safety-Dataset-2.0 into our
processed JSONL schema.

Aegis covers categories missing from our Korean datasets:
  - Violence, Criminal planning, Sexual content, Self-harm
  - Weapons, Unauthorized advice, Fraud, Malware
  - Harassment, Hate speech (English)

Maps Aegis categories to our unified topic taxonomy.

Usage:
    python fetch_aegis.py --output-dir /data/processed_split
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


# Map Aegis violated_categories to our topic taxonomy
AEGIS_TOPIC_MAP = {
    "Criminal Planning/Confessions": "criminal_planning",
    "Violence": "violence",
    "Hate/Identity Hate": "hate_speech",
    "Harassment": "harassment",
    "Profanity": "profanity",
    "Sexual": "sexual_content",
    "Controlled/Regulated Substances": "drugs",
    "PII/Privacy": "privacy",
    "Guns and Illegal Weapons": "weapons",
    "Suicide and Self Harm": "self_harm",
    "Unauthorized Advice": "unauthorized_advice",
    "Political/Misinformation/Conspiracy": "misinformation",
    "Fraud/Deception": "fraud",
    "Sexual (minor)": "child_exploitation",
    "Immoral/Unethical": "harmful_content",
    "Illegal Activity": "illegal",
    "Threat": "violence",
    "Malware": "malware",
    "Copyright/Trademark/Plagiarism": "copyright",
    "High Risk Gov Decision Making": "unauthorized_advice",
    "Manipulation": "manipulation",
    "Needs Caution": "needs_caution",
    "Other": "other",
}

# Map topics to guardrail type
def topic_to_type(topics):
    if not topics:
        return "safety-classifier"
    safety_topics = {"criminal_planning", "violence", "weapons", "self_harm",
                     "child_exploitation", "illegal", "malware"}
    moderation_topics = {"hate_speech", "harassment", "profanity", "sexual_content"}
    pii_topics = {"privacy"}
    output_topics = {"misinformation", "unauthorized_advice", "fraud", "copyright"}

    for t in topics:
        if t in pii_topics:
            return "pii-filter"
        if t in output_topics:
            return "output-validation"
        if t in moderation_topics:
            return "moderation"
        if t in safety_topics:
            return "safety-classifier"
    return "safety-classifier"


def convert_record(row):
    """Convert an Aegis record to our schema.

    Uses prompt_label for classification. If prompt is REDACTED, skip.
    """
    prompt = (row.get("prompt") or "").strip()
    if not prompt or prompt == "REDACTED":
        return None

    prompt_label = (row.get("prompt_label") or "").strip().lower()
    blocked = prompt_label == "unsafe"

    violated = (row.get("violated_categories") or "").strip()
    topics = []
    if violated:
        for cat in violated.split(","):
            cat = cat.strip()
            mapped = AEGIS_TOPIC_MAP.get(cat)
            if mapped and mapped not in ("needs_caution", "other"):
                topics.append(mapped)

    # Deduplicate preserving order
    topics = list(dict.fromkeys(topics))
    guardrail_type = topic_to_type(topics) if blocked else "safety-classifier"

    # Safe records: clear topics
    if not blocked:
        topics = []

    return {
        "query": prompt,
        "blocked": blocked,
        "type": guardrail_type,
        "answer": [],
        "topic": topics,
    }


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/processed_split")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading nvidia/Aegis-AI-Content-Safety-Dataset-2.0...")
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    # Aegis has train/validation/test splits — use them directly
    split_map = {
        "train": "train",
        "validation": "train",  # merge val into train (small)
        "test": "test",
    }

    train_records = []
    test_records = []

    for aegis_split, our_split in split_map.items():
        if aegis_split not in ds:
            continue
        for row in ds[aegis_split]:
            rec = convert_record(row)
            if rec:
                if our_split == "train":
                    train_records.append(rec)
                else:
                    test_records.append(rec)

    # Write
    name = "aegis-content-safety"
    train_path = out_dir / f"{name}.train.jsonl"
    test_path = out_dir / f"{name}.test.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    from collections import Counter
    train_topics = Counter()
    for r in train_records:
        for t in r.get("topic", []):
            train_topics[t] += 1

    n_blocked_train = sum(1 for r in train_records if r["blocked"])
    n_blocked_test = sum(1 for r in test_records if r["blocked"])

    print(f"\n  train: {len(train_records)} (blocked={n_blocked_train}, safe={len(train_records)-n_blocked_train})")
    print(f"  test:  {len(test_records)} (blocked={n_blocked_test}, safe={len(test_records)-n_blocked_test})")
    print(f"\n  Topic distribution (train):")
    for t, c in train_topics.most_common():
        print(f"    {t:<30} {c:>5}")
    print(f"\n  Saved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
