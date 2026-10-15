#!/usr/bin/env python3
"""Phase 3: fit an isotonic calibrator from (cosine_sim, label) pairs.

Reads a JSONL file of pairs (one ``{"sim": float, "label": 0|1}`` per line)
collected by ``benchmark_bantopics.py --emit-pairs`` (or any external source)
and writes a pickle that the scanner can load via ``ScannerConfig.calibrator_path``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.calibrator import IsotonicCalibrator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    sims, labels = [], []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sims.append(float(row["sim"]))
            labels.append(int(row["label"]))

    cal = IsotonicCalibrator.fit(np.array(sims), np.array(labels))
    cal.save(args.output)
    print(f"fit isotonic calibrator on {len(sims)} pairs → {args.output}")


if __name__ == "__main__":
    main()
