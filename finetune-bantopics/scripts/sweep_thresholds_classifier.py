#!/usr/bin/env python3
"""Sweep decision thresholds on the trained classifier.

Loads the merged classifier model once, encodes every eval prompt's sentences
once, gets sigmoid probabilities once. Then for each candidate threshold,
recomputes only the per-section confusion matrix — fast.

Output: JSON with full grid + a per-section table printed to stdout to find the
operating point that hits production targets.

    docker compose run --rm sweep-thresholds-classifier
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
    p.add_argument("--model", required=True, type=Path,
                   help="Trained classifier dir (merged_for_inference).")
    p.add_argument("--eval-suite", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--attn-implementation", default="eager")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from finetune_bantopics.splitter import split_sentences
    from finetune_bantopics.topics import all_topics

    topic_list = all_topics()
    topic_idx = {t: i for i, t in enumerate(topic_list)}

    rows = load_suite(args.eval_suite)
    print(f"# loaded {len(rows)} eval rows")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model),
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    # Score each row to a single max-probability across (sentences × requested topics).
    t0 = time.perf_counter()
    per_row_score = np.zeros(len(rows), dtype=np.float32)
    per_row_section = []
    per_row_expected = np.zeros(len(rows), dtype=bool)

    with torch.no_grad():
        for i, r in enumerate(rows):
            per_row_section.append(r.get("eval_section", "?"))
            per_row_expected[i] = r["expected"] == "BLOCK"

            idxs = []
            for t in r["topics"]:
                norm = t.strip().lower().replace("-", " ").replace("_", " ")
                if norm in topic_idx:
                    idxs.append(topic_idx[norm])
            if not idxs:
                per_row_score[i] = 0.0
                continue

            sentences = split_sentences(
                r["prompt"],
                language=r.get("language", "en"),
                backend="pysbd",
            ) or [r["prompt"]]
            enc = tokenizer(
                sentences, padding=True, truncation=True,
                max_length=args.max_seq_length, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits          # S × n_labels
            probs = torch.sigmoid(logits)         # S × n_labels
            per_row_score[i] = float(probs[:, idxs].max().item())

    print(f"# scored {len(rows)} prompts in {time.perf_counter()-t0:.1f}s")
    print(f"# score distribution: min={per_row_score.min():.3f}  "
          f"p10={np.percentile(per_row_score, 10):.3f}  "
          f"p50={np.percentile(per_row_score, 50):.3f}  "
          f"p90={np.percentile(per_row_score, 90):.3f}  "
          f"max={per_row_score.max():.3f}")

    sections = np.array(per_row_section)
    section_names = sorted(set(per_row_section))

    grid = []
    for thr in args.thresholds:
        actual_block = per_row_score >= thr
        for section in section_names:
            mask = sections == section
            tp = int((mask & per_row_expected & actual_block).sum())
            fn = int((mask & per_row_expected & ~actual_block).sum())
            fp = int((mask & ~per_row_expected & actual_block).sum())
            tn = int((mask & ~per_row_expected & ~actual_block).sum())
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

    # Pretty per-threshold × per-section table.
    print(f"\n# {'thr':<5} {'section':<22} {'n':>4} {'recall':>7} {'prec':>7} {'acc':>7} {'pass':>7}")
    by_thr_section = {(c["threshold"], c["section"]): c for c in grid}
    for thr in args.thresholds:
        for section in section_names:
            c = by_thr_section[(thr, section)]
            pass_str = f"{c['pass_rate']:.3f}" if c['pass_rate'] is not None else "  -  "
            print(f"  {thr:<5.2f} {section:<22} {c['n']:>4} {c['recall']:>7.3f} "
                  f"{c['precision']:>7.3f} {c['accuracy']:>7.3f} {pass_str:>7}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": str(args.model),
        "score_distribution": {
            "min": float(per_row_score.min()),
            "p10": float(np.percentile(per_row_score, 10)),
            "p50": float(np.percentile(per_row_score, 50)),
            "p90": float(np.percentile(per_row_score, 90)),
            "max": float(per_row_score.max()),
        },
        "grid": grid,
    }, indent=2))
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()
