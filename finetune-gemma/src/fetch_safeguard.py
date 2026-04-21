#!/usr/bin/env python3
"""
Download and convert the xTRam1/safe-guard-prompt-injection dataset
into our processed JSONL schema.

Input:  HuggingFace dataset (auto-downloaded)
Output: processed_split/safe-guard-prompt-injection.{train,test}.jsonl

Schema per line:
    {"query": "...", "blocked": true/false, "type": "safety-classifier",
     "answer": [], "topic": ["prompt_injection"]}
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def convert_record(row):
    """Convert a safe-guard record to our schema."""
    text = (row.get("text") or "").strip()
    if not text:
        return None
    label = row.get("label", 0)
    blocked = label == 1
    return {
        "query": text,
        "blocked": blocked,
        "type": "safety-classifier",
        "answer": [],
        "topic": ["prompt_injection"] if blocked else [],
    }


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/processed_split",
                        help="Output directory for train/test JSONL files")
    parser.add_argument("--dataset", default="xTRam1/safe-guard-prompt-injection")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset}...")
    ds = load_dataset(args.dataset)

    for split_name in ["train", "test"]:
        if split_name not in ds:
            print(f"  SKIP split '{split_name}' (not in dataset)")
            continue

        records = []
        for row in ds[split_name]:
            rec = convert_record(row)
            if rec:
                records.append(rec)

        n_blocked = sum(1 for r in records if r["blocked"])
        out_path = out_dir / f"safe-guard-prompt-injection.{split_name}.jsonl"
        write_jsonl(records, out_path)
        print(f"  {split_name}: {len(records)} records (blocked={n_blocked}, safe={len(records)-n_blocked}) → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
