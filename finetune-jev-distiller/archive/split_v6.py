"""Split train_v6.jsonl by whole source: held-out sources go to validation and never to training.

    python split_v6.py --input data/train_v6.jsonl --train data/v6_train.jsonl --val data/v6_val.jsonl

A random slice of the training sources would share their distribution and say little about unseen data, so the
checkpoint is chosen on whole datasets the model never trains on, one per kind, with their own (gold) labels:
CategoricalHarmfulQA (plain harmful requests), deepset (prompt injection, both labels), CoCoNot (benign prompts
that sound sensitive) and Aegis 2.0 responses (response moderation, both labels). Each is capped at --per-source
rows (label-balanced where it has both) to keep evaluation short; the rest of a held-out source is unused.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

HOLDOUT = ("categorical_harmfulqa", "deepset_injection", "coconot", "aegis_responses")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--per-source", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train, held = [], defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(args.input)):
        if r["source"] in HOLDOUT:
            held[r["source"]][r["gold_block"]].append(r)
        else:
            train.append(r)
    val = []
    for source in HOLDOUT:
        by_label = held[source]
        quota = args.per_source // len(by_label)
        for label in sorted(by_label):
            rows = by_label[label]
            val += rng.sample(rows, min(quota, len(rows)))

    for path, rows in ((args.train, train), (args.val, val)):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train {len(train)} -> {args.train}")
    print(f"val {len(val)} -> {args.val}: {dict(Counter((r['source'], r['gold_block']) for r in val))}")


if __name__ == "__main__":
    main()
