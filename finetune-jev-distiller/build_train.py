"""Draw training candidates from each dataset's train splits, with the dataset's own block/allow label.

    python build_train.py --per-dataset 2000 --exclude data/benchmark_candidates.jsonl --output data/train_candidates.jsonl

Benchmark-only datasets have no train splits and are never drawn. Any prompt in --exclude (every benchmark
candidate, verified or disputed) is left out, so no benchmark prompt can reach training even when a dataset's
train and test splits overlap. Candidates are verified with verify_glm.py, like the benchmark.
"""

import argparse
import json
import logging
import random
from collections import Counter

from datasets_spec import DATASETS, load_rows

log = logging.getLogger("train-data")


def main():
    trainable = [n for n, s in DATASETS.items() if s["train"]]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=trainable, help=f"any of {trainable}")
    ap.add_argument("--per-dataset", type=int, default=2000)
    ap.add_argument("--exclude", nargs="+", required=True, help="jsonl files whose prompts must not be trained on")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    unknown = set(args.datasets) - set(trainable)
    if unknown:
        raise SystemExit(f"not trainable: {sorted(unknown)}; choose from {trainable}")

    seen = {json.loads(l)["text"].strip() for path in args.exclude for l in open(path)}
    excluded = len(seen)
    rng = random.Random(args.seed)
    out, overlap = [], Counter()
    for name in args.datasets:
        kept = []
        for row in load_rows(name, "train", args.per_dataset, rng):
            if len(kept) == args.per_dataset:
                break
            if row["text"] in seen:
                overlap[name] += 1
                continue
            seen.add(row["text"])
            kept.append(row)
        out += kept
        c = Counter(r["gold_block"] for r in kept)
        log.info(f"{name:<26} {len(kept):>5}  block={c[True]} allow={c[False]}  (skipped {overlap[name]} seen)")

    with open(args.output, "w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    c = Counter(r["gold_block"] for r in out)
    log.info(f"wrote {len(out)} candidates (block={c[True]} allow={c[False]}), none of the {excluded} excluded "
             f"prompts -> {args.output}")


if __name__ == "__main__":
    main()
