"""Draw benchmark candidates from each dataset's benchmark split, with the dataset's own block/allow label.

    python build_benchmark.py --per-dataset 300 --output data/benchmark_candidates.jsonl

Candidates still need verification (verify_glm.py) before they form the benchmark.
"""

import argparse
import json
import logging
import random
from collections import Counter

from datasets_spec import DATASETS, load_rows

log = logging.getLogger("bench")


def main():
    benchable = [n for n, s in DATASETS.items() if s["bench"]]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=benchable, help=f"any of {benchable}")
    ap.add_argument("--per-dataset", type=int, default=300)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    unknown = set(args.datasets) - set(benchable)
    if unknown:
        raise SystemExit(f"not benchmarkable: {sorted(unknown)}; choose from {benchable}")

    rng = random.Random(args.seed)
    seen, out = set(), []
    for name in args.datasets:
        kept = []
        for row in load_rows(name, "bench", args.per_dataset, rng):
            if len(kept) == args.per_dataset:
                break
            if row["text"] not in seen:
                seen.add(row["text"])
                kept.append(row)
        out += kept
        c = Counter(r["gold_block"] for r in kept)
        log.info(f"{name:<20} {len(kept):>4}  block={c[True]} allow={c[False]}")

    with open(args.output, "w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"wrote {len(out)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
