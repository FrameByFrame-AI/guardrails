#!/usr/bin/env python3
"""
Download and convert public prompt-injection / jailbreak datasets into our
processed JSONL schema.

Datasets fetched:
  1. deepset/prompt-injections
  2. JailbreakBench/JBB-Behaviors
  3. JailbreakV-28K/JailBreakV-28k
  4. DhruvTre/jailbreakbench-paraphrase-2025-08
  5. microsoft/llmail-inject-challenge
  6. hendzh/PromptShield
  7. geekyrakshit/prompt-injection-dataset

Output per dataset: processed_split/<name>.{train,test}.jsonl

Usage:
    python fetch_injection_datasets.py --output-dir /data/processed_split
    python fetch_injection_datasets.py --output-dir /data/processed_split --only deepset-prompt-injections
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42
TEST_FRACTION = 0.1


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_and_write(records, name, out_dir, seed=SEED):
    rng = random.Random(seed + hash(name) & 0x7FFFFFFF)

    blocked = [r for r in records if r["blocked"]]
    safe = [r for r in records if not r["blocked"]]
    rng.shuffle(blocked)
    rng.shuffle(safe)

    n_b_test = int(round(len(blocked) * TEST_FRACTION))
    n_s_test = int(round(len(safe) * TEST_FRACTION))

    test = blocked[:n_b_test] + safe[:n_s_test]
    train = blocked[n_b_test:] + safe[n_s_test:]
    rng.shuffle(test)
    rng.shuffle(train)

    train_path = out_dir / f"{name}.train.jsonl"
    test_path = out_dir / f"{name}.test.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)

    n_blocked_train = sum(1 for r in train if r["blocked"])
    n_blocked_test = sum(1 for r in test if r["blocked"])
    print(f"  {name}")
    print(f"    train: {len(train)} (blocked={n_blocked_train}, safe={len(train)-n_blocked_train})")
    print(f"    test:  {len(test)} (blocked={n_blocked_test}, safe={len(test)-n_blocked_test})")
    return len(train) + len(test)


def make_record(text, blocked, topic=None):
    text = (text or "").strip()
    if not text:
        return None
    return {
        "query": text,
        "blocked": blocked,
        "type": "safety-classifier",
        "answer": [],
        "topic": [topic or "prompt_injection"] if blocked else [],
    }


# ---------------------------------------------------------------------------
# Per-dataset converters

def fetch_deepset_prompt_injections(out_dir):
    """deepset/prompt-injections — binary (0=safe, 1=injection)"""
    ds = load_dataset("deepset/prompt-injections")
    records = []
    for row in ds["train"]:
        rec = make_record(row.get("text"), blocked=(row.get("label") == 1))
        if rec:
            records.append(rec)
    return split_and_write(records, "deepset-prompt-injections", out_dir)


def fetch_jbb_behaviors(out_dir):
    """JailbreakBench/JBB-Behaviors — all are harmful behaviors (blocked)"""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    records = []
    for split in ds:
        for row in ds[split]:
            text = row.get("Behavior") or row.get("behavior") or row.get("Goal") or row.get("goal") or ""
            rec = make_record(text, blocked=True, topic="jailbreak")
            if rec:
                records.append(rec)
    # Deduplicate
    seen = set()
    unique = []
    for r in records:
        if r["query"] not in seen:
            seen.add(r["query"])
            unique.append(r)
    return split_and_write(unique, "jbb-behaviors", out_dir)


def fetch_jailbreakv_28k(out_dir):
    """JailbreakV-28K/JailBreakV-28k — large jailbreak corpus"""
    ds = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")
    records = []
    for split in ds:
        for row in ds[split]:
            text = row.get("jailbreak_query") or row.get("query") or row.get("prompt") or row.get("text") or ""
            rec = make_record(text, blocked=True, topic="jailbreak")
            if rec:
                records.append(rec)
    seen = set()
    unique = []
    for r in records:
        if r["query"] not in seen:
            seen.add(r["query"])
            unique.append(r)
    return split_and_write(unique, "jailbreakv-28k", out_dir)


def fetch_jailbreakbench_paraphrase(out_dir):
    """DhruvTre/jailbreakbench-paraphrase-2025-08 — paraphrased jailbreak variants"""
    try:
        ds = load_dataset("DhruvTre/jailbreakbench-paraphrase-2025-08")
    except Exception as e:
        print(f"  SKIP jailbreakbench-paraphrase: {e}")
        return 0
    records = []
    for split in ds:
        for row in ds[split]:
            text = row.get("paraphrase") or row.get("text") or row.get("prompt") or ""
            rec = make_record(text, blocked=True, topic="jailbreak")
            if rec:
                records.append(rec)
    seen = set()
    unique = []
    for r in records:
        if r["query"] not in seen:
            seen.add(r["query"])
            unique.append(r)
    return split_and_write(unique, "jailbreakbench-paraphrase", out_dir)


def fetch_llmail_inject(out_dir, max_samples=5000):
    """microsoft/llmail-inject-challenge — indirect injection in emails.

    All records are injection attempts embedded in email bodies. The 'body'
    field contains the email text with hidden prompt injection instructions.
    We sample to avoid overwhelming the training set (~460k total).
    """
    try:
        ds = load_dataset("microsoft/llmail-inject-challenge")
    except Exception as e:
        print(f"  SKIP llmail-inject: {e}")
        return 0
    records = []
    for split in ds:
        for row in ds[split]:
            body = (row.get("body") or "").strip()
            subject = (row.get("subject") or "").strip()
            if not body:
                continue
            # Combine subject + body as the full email text to classify
            text = f"Subject: {subject}\n\n{body}" if subject else body
            rec = make_record(text, blocked=True, topic="indirect_injection")
            if rec:
                records.append(rec)

    # Sample to manageable size
    rng = random.Random(SEED)
    if len(records) > max_samples:
        rng.shuffle(records)
        records = records[:max_samples]
        print(f"    (sampled {max_samples} from {len(records)} total)")

    return split_and_write(records, "llmail-inject-challenge", out_dir)


def fetch_promptshield(out_dir):
    """hendzh/PromptShield — prompt injection detection"""
    try:
        ds = load_dataset("hendzh/PromptShield")
    except Exception as e:
        print(f"  SKIP PromptShield: {e}")
        return 0
    records = []
    for split in ds:
        for row in ds[split]:
            text = row.get("text") or row.get("prompt") or ""
            label = row.get("label", 0)
            blocked = label == 1 or str(label).lower() in ("true", "injection", "malicious")
            rec = make_record(text, blocked=blocked)
            if rec:
                records.append(rec)
    return split_and_write(records, "promptshield", out_dir)


def fetch_geekyrakshit(out_dir, max_samples=10000):
    """geekyrakshit/prompt-injection-dataset — large; sample to avoid dominating training."""
    try:
        ds = load_dataset("geekyrakshit/prompt-injection-dataset")
    except Exception as e:
        print(f"  SKIP geekyrakshit: {e}")
        return 0
    records = []
    for split in ds:
        for row in ds[split]:
            text = row.get("text") or row.get("prompt") or ""
            label = row.get("label", 0)
            blocked = label == 1 or str(label).lower() in ("true", "injection", "malicious")
            rec = make_record(text, blocked=blocked)
            if rec:
                records.append(rec)

    if len(records) > max_samples:
        rng = random.Random(SEED)
        blocked = [r for r in records if r["blocked"]]
        safe = [r for r in records if not r["blocked"]]
        ratio = len(blocked) / len(records)
        n_b = round(max_samples * ratio)
        n_s = max_samples - n_b
        rng.shuffle(blocked)
        rng.shuffle(safe)
        records = blocked[:n_b] + safe[:n_s]
        rng.shuffle(records)
        print(f"    (sampled {len(records)} from {len(blocked)+len(safe)} total)")

    return split_and_write(records, "geekyrakshit-prompt-injection", out_dir)


# ---------------------------------------------------------------------------

FETCHERS = {
    "deepset-prompt-injections": fetch_deepset_prompt_injections,
    "jbb-behaviors": fetch_jbb_behaviors,
    "jailbreakv-28k": fetch_jailbreakv_28k,
    "jailbreakbench-paraphrase": fetch_jailbreakbench_paraphrase,
    "llmail-inject-challenge": fetch_llmail_inject,
    "promptshield": fetch_promptshield,
    "geekyrakshit-prompt-injection": fetch_geekyrakshit,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/processed_split")
    parser.add_argument("--only", default="", help="Comma-separated list of datasets to fetch (empty=all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = FETCHERS
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        targets = {k: v for k, v in FETCHERS.items() if k in names}

    print(f"Fetching {len(targets)} datasets → {out_dir}\n")
    total = 0
    for name, fetcher in targets.items():
        try:
            n = fetcher(out_dir)
            total += n
        except Exception as e:
            print(f"  ERROR {name}: {e}")
    print(f"\nTotal: {total} records across {len(targets)} datasets")


if __name__ == "__main__":
    main()
