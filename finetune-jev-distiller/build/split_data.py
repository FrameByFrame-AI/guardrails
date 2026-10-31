"""Split gliner records into train / val / test by a hash of each prompt's text.

    python split_data.py --input data/gliner_train.jsonl --manifest data/train_verified.jsonl --out-prefix data/split --test 0

Writes <prefix>_train.jsonl, <prefix>_val.jsonl (checkpoint selection) and <prefix>_test.jsonl (reporting only).
A prompt's split depends only on its text and --salt, so it is independent of file order and a prompt keeps its
split when the dataset grows (no test prompt later leaks into training). Proportions are approximate on small
sets. The per-source counts printed come from the manifest.
"""

import argparse
import hashlib
import json
from collections import Counter


def bucket(text: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}\0{text}".encode()).hexdigest()[:12], 16) / 16 ** 12


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="gliner records from to_gliner.py")
    ap.add_argument("--manifest", required=True, help="rows with text and source, for per-source counts")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--salt", default="jev-distiller-v1", help="change only to deliberately reshuffle all splits")
    args = ap.parse_args()

    source = {}
    for line in open(args.manifest):
        row = json.loads(line)
        source[row["text"].strip()] = row["source"]

    splits = {"train": [], "val": [], "test": []}
    per_source = {k: Counter() for k in splits}
    for line in open(args.input):
        rec = json.loads(line)
        b = bucket(rec["input"], args.salt)
        split = "test" if b < args.test else "val" if b < args.test + args.val else "train"
        splits[split].append(rec)
        per_source[split][source.get(rec["input"], "unknown")] += 1

    for split, recs in splits.items():
        recs.sort(key=lambda r: bucket(r["input"], args.salt + split))  # stable order, not grouped by source
        with open(f"{args.out_prefix}_{split}.jsonl", "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in splits.items()}, "->", f"{args.out_prefix}_{{train,val,test}}.jsonl")
    for split in splits:
        print(f"  {split}: {dict(sorted(per_source[split].items()))}")


if __name__ == "__main__":
    main()
