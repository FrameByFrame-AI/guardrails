#!/usr/bin/env python3
"""Sweep decision thresholds on the full eval suite, per section.

Loads a trained scanner once, encodes every prompt's sentences once, encodes
every requested topic once (cached), then for each candidate threshold
recomputes only the decision boundary — fast.

For each (threshold × eval_section), reports accuracy, recall, precision, and
pass_rate (the metric that matters for xstest_safe). Writes a JSON with the
full grid plus per-section "best threshold" picks.

    docker compose run --rm sweep-thresholds-on-suite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.scanner import (
    BanTopicsScanner,
    ScannerConfig,
    DEFAULT_MODEL,
    _pool,
)
from finetune_bantopics.splitter import split_sentences


def load_suite(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Trained model dir (e.g. /models/.../merged_for_inference) or HF id.")
    p.add_argument("--eval-suite", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--topic-template", default="This text is about {topic}.")
    p.add_argument("--pooling", default="max", choices=["max", "top3-mean"])
    p.add_argument("--splitter-backend", default="pysbd")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    rows = load_suite(args.eval_suite)
    print(f"# loaded {len(rows)} eval rows from {args.eval_suite}")

    cfg = ScannerConfig(
        model_name_or_path=args.model,
        topic_template=args.topic_template,
        pooling=args.pooling,
        splitter_backend=args.splitter_backend,
        device=args.device,
    )
    scanner = BanTopicsScanner(cfg)

    t0 = time.perf_counter()
    flat_sentences: list[str] = []
    sentence_offsets = [0]
    for r in rows:
        sents = split_sentences(
            r["prompt"],
            language=r.get("language", "en"),
            backend=args.splitter_backend,
        ) or [r["prompt"]]
        flat_sentences.extend(sents)
        sentence_offsets.append(len(flat_sentences))

    sent_vecs = scanner.encode(flat_sentences)
    print(f"# encoded {len(flat_sentences)} sentences in {time.perf_counter()-t0:.1f}s")

    # Per-row max-cosine score across that row's *requested* topics.
    t0 = time.perf_counter()
    per_row_max_sim = np.zeros(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        topic_vecs = scanner._encode_topics(r["topics"])
        sims = sent_vecs[sentence_offsets[i]:sentence_offsets[i + 1]] @ topic_vecs.T
        if sims.size == 0:
            per_row_max_sim[i] = 0.0
            continue
        per_topic = _pool(sims, mode=args.pooling)
        per_row_max_sim[i] = float(per_topic.max())
    print(f"# scored {len(rows)} prompts in {time.perf_counter()-t0:.1f}s "
          f"(p50 sim={float(np.median(per_row_max_sim)):.3f}, "
          f"min={float(per_row_max_sim.min()):.3f}, "
          f"max={float(per_row_max_sim.max()):.3f})")

    expected_block = np.array([r["expected"] == "BLOCK" for r in rows], dtype=bool)
    sections = np.array([r.get("eval_section", "?") for r in rows])

    grid = []
    for thr in args.thresholds:
        actual_block = per_row_max_sim >= thr
        for section in sorted(set(sections.tolist())):
            mask = sections == section
            tp = int((mask & expected_block & actual_block).sum())
            fn = int((mask & expected_block & ~actual_block).sum())
            fp = int((mask & ~expected_block & actual_block).sum())
            tn = int((mask & ~expected_block & ~actual_block).sum())
            n = tp + fp + tn + fn
            grid.append({
                "threshold": thr,
                "section": section,
                "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "accuracy": (tp + tn) / max(n, 1),
                "pass_rate": tn / max(tn + fp, 1) if (tn + fp) else None,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "topic_template": args.topic_template,
        "pooling": args.pooling,
        "score_distribution": {
            "p10": float(np.percentile(per_row_max_sim, 10)),
            "p50": float(np.percentile(per_row_max_sim, 50)),
            "p90": float(np.percentile(per_row_max_sim, 90)),
            "min": float(per_row_max_sim.min()),
            "max": float(per_row_max_sim.max()),
        },
        "grid": grid,
    }, indent=2))

    # Per-section table, sorted by threshold for readability.
    print(f"\n# {'thr':<5} {'section':<22} {'recall':>7} {'prec':>7} {'acc':>7} {'pass':>7}")
    by_thr_section = {(c["threshold"], c["section"]): c for c in grid}
    for thr in args.thresholds:
        for section in sorted(set(sections.tolist())):
            c = by_thr_section[(thr, section)]
            pass_str = f"{c['pass_rate']:.3f}" if c['pass_rate'] is not None else "  -  "
            print(f"  {thr:<5.2f} {section:<22} {c['recall']:>7.3f} {c['precision']:>7.3f} "
                  f"{c['accuracy']:>7.3f} {pass_str:>7}")
        print()
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()
