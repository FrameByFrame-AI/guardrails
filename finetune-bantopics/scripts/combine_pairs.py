#!/usr/bin/env python3
"""Merge per-source pair JSONLs into one training file.

Concatenates inputs, dedupes by (sentence, topic) — same sentence emitted by
two sources for the same topic counts once, but the same sentence for two
*different* topics is kept as separate rows. Optionally caps per-topic to
prevent the long tail (e.g. ``illegal activity`` from Aegis) from drowning
out under-represented security topics.

    docker compose run --rm combine-pairs
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cap-per-topic", type=int, default=None,
                   help="If set, downsample each topic to at most N rows.")
    p.add_argument("--seed", type=int, default=20260430)
    args = p.parse_args()

    seen: set[tuple[str, str]] = set()
    by_topic: dict[str, list[dict]] = defaultdict(list)
    by_source: Counter = Counter()
    n_input = 0

    for path in args.inputs:
        if not path.exists():
            print(f"# skip {path} (not found)")
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
                n_input += 1
                key = (sentence, topic)
                if key in seen:
                    continue
                seen.add(key)
                by_topic[topic].append(row)
                by_source[row.get("source", "?")] += 1

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    capped: Counter = Counter()
    for topic, rows in by_topic.items():
        if args.cap_per_topic and len(rows) > args.cap_per_topic:
            rng.shuffle(rows)
            capped[topic] = len(rows) - args.cap_per_topic
            rows = rows[: args.cap_per_topic]
        out_rows.extend(rows)

    rng.shuffle(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for row in out_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"# read {n_input} rows, wrote {len(out_rows)} unique (sentence, topic) pairs → {args.output}")
    print(f"\n# per source:")
    for s, n in by_source.most_common():
        print(f"  {n:7d}  {s}")
    print(f"\n# per topic (post-cap):")
    final_topic = Counter(r["topic"] for r in out_rows)
    for t, n in final_topic.most_common():
        suffix = f"  (capped from {final_topic[t] + capped[t]})" if capped[t] else ""
        print(f"  {n:7d}  {t}{suffix}")


if __name__ == "__main__":
    main()
