#!/usr/bin/env python3
"""Collapse (sentence, topic) pairs into multi-label rows for classifier training.

Input  (one or more pair files):
    {"sentence": "...", "topic": "phishing", ...}
    {"sentence": "...", "topic": "fraud",    ...}
    {"sentence": "(same sentence)", "topic": "social engineering", ...}

Output (one merged JSONL):
    {"sentence": "...", "labels": [0,0,...,1,1,...,1,...,0]}    # 24-d binary

Label index alignment: ``finetune_bantopics.topics.all_topics()`` order. Same
order is consumed by ``train_classifier.py`` and the ClassifierBanTopicsScanner.

    docker compose run --rm collapse-pairs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.topics import all_topics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    topic_list = all_topics()
    topic_to_idx = {t: i for i, t in enumerate(topic_list)}
    print(f"# label space: {len(topic_list)} topics")
    for i, t in enumerate(topic_list):
        print(f"   {i:2d}  {t}")

    sentence_topics: dict[str, set[int]] = defaultdict(set)
    n_in = 0
    skipped_unknown = Counter()
    for path in args.inputs:
        if not path.exists():
            print(f"# skip {path} (not found)", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sentence = row.get("sentence")
                topic = row.get("topic")
                if not sentence or not topic:
                    continue
                n_in += 1
                idx = topic_to_idx.get(topic)
                if idx is None:
                    skipped_unknown[topic] += 1
                    continue
                sentence_topics[sentence].add(idx)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_count = Counter()
    multi_label_rows = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for sent, idxs in sentence_topics.items():
            labels = [0.0] * len(topic_list)
            for i in idxs:
                labels[i] = 1.0
                label_count[topic_list[i]] += 1
            if len(idxs) > 1:
                multi_label_rows += 1
            out.write(json.dumps({"sentence": sent, "labels": labels}, ensure_ascii=False) + "\n")

    n_out = len(sentence_topics)
    print(f"\n# read {n_in} pairs, wrote {n_out} unique sentences ({multi_label_rows} multi-label) → {args.output}")
    print(f"# avg labels/sentence = {sum(label_count.values()) / max(n_out, 1):.2f}\n")
    for t in topic_list:
        print(f"  {label_count[t]:7d}  {t}")
    if skipped_unknown:
        print("\n# skipped unknown topics (not in label space):")
        for t, n in skipped_unknown.most_common():
            print(f"  {n:6d}  {t}")


if __name__ == "__main__":
    main()
