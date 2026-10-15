#!/usr/bin/env python3
"""Extract the BanTopics rows from the cross-scanner benchmark CSV into a JSONL fixture.

The source CSV (``scanner_benchmark_results.csv`` at the repo root) holds 50
BanTopics prompts with expected_decision in {BLOCK, PASS}. The CSV does not
record the topic list; it is hard-coded to the same set the existing scanner
benchmark uses (matching ``request_payloads.json``).

    python scripts/extract_bantopics_eval.py \\
        --source ../scanner_benchmark_results.csv \\
        --output data/eval/bantopics_50.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_TOPICS = ["ransomware", "phishing", "malware", "cyber attack"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    p.add_argument("--language", default="en")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_block = n_pass = 0
    with open(args.source, newline="", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            if row["scanner"] != "BanTopics":
                continue
            label = row["expected_decision"].strip().upper()
            if label not in {"BLOCK", "PASS"}:
                continue
            record = {
                "prompt": row["prompt"],
                "expected": label,
                "topics": list(args.topics),
                "language": args.language,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_block += int(label == "BLOCK")
            n_pass += int(label == "PASS")

    print(f"wrote {args.output}: {n_block} BLOCK, {n_pass} PASS")


if __name__ == "__main__":
    main()
