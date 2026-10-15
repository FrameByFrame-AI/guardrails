#!/usr/bin/env python3
"""Phase 0 benchmark for BanTopics.

Two modes:

* ``--baseline`` runs the legacy DeBERTa NLI zero-shot model. Uses a
  ``datasets.Dataset`` + ``KeyDataset`` to feed the HF pipeline in batches
  so the GPU is saturated instead of one prompt at a time.

* default runs the Granite bi-encoder sweep over (threshold, template,
  pooling). The sweep is fully vectorised: encode every sentence and every
  topic-template pair once, then run the whole grid as numpy ops on the
  pre-computed similarity matrix.

Per-request latency (p50 / p95) is reported separately from sweep accuracy.
Sweep cells use throughput-style timing; latency mode (``--latency-pass``)
runs prompts one at a time after a warmup, which is what production sees.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.scanner import (
    BanTopicsScanner,
    ScanRequest,
    ScannerConfig,
    DEFAULT_MODEL,
    _pool,
)
from finetune_bantopics.splitter import split_sentences


SWEEP_THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
SWEEP_TEMPLATES = [
    "{topic}",
    "Text discussing {topic}.",
    "This text is about {topic}.",
]
SWEEP_POOLINGS = ["max", "top3-mean"]


def load_eval(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confusion(expected_block: np.ndarray, actual_block: np.ndarray) -> dict:
    tp = int(((expected_block) & (actual_block)).sum())
    fn = int(((expected_block) & (~actual_block)).sum())
    fp = int(((~expected_block) & (actual_block)).sum())
    tn = int(((~expected_block) & (~actual_block)).sum())
    n = tp + fp + tn + fn
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "accuracy": (tp + tn) / max(n, 1),
    }


def percentiles(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    s = sorted(latencies)
    return {
        "p50_ms": median(s),
        "p95_ms": s[min(int(len(s) * 0.95), len(s) - 1)],
    }


# ---------------------------------------------------------------------------
# baseline: legacy DeBERTa NLI zero-shot, batched via datasets.Dataset
# ---------------------------------------------------------------------------

BASELINE_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"


def run_baseline(
    eval_path: Path,
    output_path: Path,
    threshold: float,
    batch_size: int,
    latency_pass: bool,
    device: int,
) -> None:
    from datasets import Dataset
    from transformers import pipeline
    from transformers.pipelines.pt_utils import KeyDataset

    rows = load_eval(eval_path)
    if not rows:
        raise SystemExit(f"no rows in {eval_path}")

    topics = rows[0]["topics"]
    if not all(r["topics"] == topics for r in rows):
        raise SystemExit("baseline path assumes a single topic list across rows")

    pipe = pipeline(
        "zero-shot-classification",
        model=BASELINE_MODEL,
        device=device,
    )

    ds = Dataset.from_list([{"prompt": r["prompt"]} for r in rows])

    # Throughput pass — let the pipeline batch on GPU.
    t0 = time.perf_counter()
    outputs = list(
        pipe(
            KeyDataset(ds, "prompt"),
            candidate_labels=topics,
            batch_size=batch_size,
            multi_label=True,
        )
    )
    throughput_s = time.perf_counter() - t0

    max_scores = np.array([max(o["scores"]) for o in outputs], dtype=np.float32)
    expected_block = np.array([r["expected"] == "BLOCK" for r in rows], dtype=bool)

    # Sweep DeBERTa thresholds for context (cheap once probabilities are computed).
    sweep = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sweep.append({"threshold": thr, **confusion(expected_block, max_scores >= thr)})

    actual_block = max_scores >= threshold
    cm = confusion(expected_block, actual_block)

    # Per-request latency from a separate sequential pass.
    lat = {"p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    if latency_pass:
        # Warmup
        for _ in range(2):
            pipe(rows[0]["prompt"], candidate_labels=topics, multi_label=True)
        latencies = []
        for r in rows:
            t = time.perf_counter()
            pipe(r["prompt"], candidate_labels=topics, multi_label=True)
            latencies.append((time.perf_counter() - t) * 1000)
        lat = {"n": len(latencies), **percentiles(latencies)}

    result = {
        "model": BASELINE_MODEL,
        "threshold": threshold,
        "batch_size": batch_size,
        "throughput": {
            "total_s": throughput_s,
            "ms_per_prompt": (throughput_s / len(rows)) * 1000,
        },
        "latency_sequential": lat,
        **cm,
        "threshold_sweep": sweep,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Granite sweep: vectorised — encode once, run grid as numpy ops
# ---------------------------------------------------------------------------


def _split_all(prompts: list[str], language: str, backend: str) -> tuple[list[str], np.ndarray]:
    """Returns (flat_sentences, offsets) where offsets[i:i+1] selects prompt i's rows."""
    flat: list[str] = []
    counts: list[int] = []
    for p in prompts:
        sents = split_sentences(p, language=language, backend=backend) or [p]
        flat.extend(sents)
        counts.append(len(sents))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return flat, offsets


def _pool_per_prompt(sims: np.ndarray, offsets: np.ndarray, mode: str) -> np.ndarray:
    """sims: N_total x T  →  per_prompt: N_prompts x T pooled by ``mode``."""
    out = np.empty((len(offsets) - 1, sims.shape[1]), dtype=np.float32)
    for i in range(len(offsets) - 1):
        block = sims[offsets[i]:offsets[i + 1]]
        if block.shape[0] == 0:
            out[i] = 0.0
        else:
            out[i] = _pool(block, mode=mode)
    return out


def run_sweep(
    eval_path: Path,
    output_path: Path,
    model: str,
    splitter_backend: str,
    splitter_language: str,
    device: str | None,
    latency_pass: bool,
    encode_batch_size: int,
) -> None:
    rows = load_eval(eval_path)
    if not rows:
        raise SystemExit(f"no rows in {eval_path}")
    topics = rows[0]["topics"]
    if not all(r["topics"] == topics for r in rows):
        raise SystemExit("sweep path assumes a single topic list across rows")

    expected_block = np.array([r["expected"] == "BLOCK" for r in rows], dtype=bool)

    # Build the scanner once — the encoder model load is the slowest single op.
    cfg = ScannerConfig(
        model_name_or_path=model,
        splitter_backend=splitter_backend,
        splitter_language=splitter_language,
        device=device,
        encode_batch_size=encode_batch_size,
    )
    scanner = BanTopicsScanner(cfg)

    # 1) Split + encode every sentence in one big batch.
    t0 = time.perf_counter()
    flat_sentences, offsets = _split_all(
        [r["prompt"] for r in rows],
        language=splitter_language,
        backend=splitter_backend,
    )
    split_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sent_vecs = scanner.encode(flat_sentences)              # N_total x D
    sent_encode_s = time.perf_counter() - t0

    print(
        f"# {len(rows)} prompts → {len(flat_sentences)} sentences | "
        f"split {split_s*1000:.1f}ms, encode {sent_encode_s*1000:.1f}ms"
    )

    # 2) For each template: encode topics once, sweep (pooling, threshold) as numpy.
    cells: list[dict] = []
    for template in SWEEP_TEMPLATES:
        topic_inputs = [template.format(topic=t) for t in topics]
        t0 = time.perf_counter()
        topic_vecs = scanner.encode(topic_inputs)            # T x D
        topic_encode_ms = (time.perf_counter() - t0) * 1000

        sims = sent_vecs @ topic_vecs.T                      # N_total x T

        for pooling in SWEEP_POOLINGS:
            per_prompt = _pool_per_prompt(sims, offsets, mode=pooling)
            max_per_prompt = per_prompt.max(axis=1)          # N_prompts
            for thr in SWEEP_THRESHOLDS:
                actual_block = max_per_prompt >= thr
                cells.append({
                    "threshold": thr,
                    "template": template,
                    "pooling": pooling,
                    "topic_encode_ms": topic_encode_ms,
                    **confusion(expected_block, actual_block),
                })

    cells.sort(key=lambda c: (-c["accuracy"], -c["recall"], c["p50_ms"] if "p50_ms" in c else 0))
    best = cells[0]
    print(
        f"\n# best cell: thr={best['threshold']} pool={best['pooling']} "
        f"tmpl={best['template']!r} acc={best['accuracy']:.3f} "
        f"rec={best['recall']:.3f} prec={best['precision']:.3f}"
    )

    # 3) End-to-end per-request latency for the best cell (production-shaped).
    latency = {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0}
    if latency_pass:
        cfg.topic_template = best["template"]
        cfg.pooling = best["pooling"]
        scanner_for_lat = BanTopicsScanner(cfg)
        # Warmup
        for _ in range(2):
            scanner_for_lat.scan(_make_request(rows[0], best["threshold"]))
        latencies = []
        for r in rows:
            t = time.perf_counter()
            scanner_for_lat.scan(_make_request(r, best["threshold"]))
            latencies.append((time.perf_counter() - t) * 1000)
        latency = {"n": len(latencies), **percentiles(latencies)}
        print(f"# latency p50={latency['p50_ms']:.1f}ms p95={latency['p95_ms']:.1f}ms")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "model": model,
        "splitter_backend": splitter_backend,
        "splitter_language": splitter_language,
        "n_prompts": len(rows),
        "n_sentences": len(flat_sentences),
        "split_ms": split_s * 1000,
        "sentence_encode_ms": sent_encode_s * 1000,
        "best": best,
        "latency_sequential": latency,
        "cells": cells,
    }, indent=2))
    print(f"wrote {output_path}")


def _make_request(row: dict, threshold: float) -> ScanRequest:
    return ScanRequest(
        scanner_name="BanTopics",
        scanner_type="prompt",
        scanner_params={"threshold": threshold, "topics": row["topics"]},
        prompt=row["prompt"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--splitter-backend", default="pysbd")
    p.add_argument("--splitter-language", default="en")
    p.add_argument("--device", default=None)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--baseline", action="store_true",
                   help="Run the legacy DeBERTa NLI baseline instead of the Granite sweep.")
    p.add_argument("--baseline-threshold", type=float, default=0.8,
                   help="Decision threshold for the baseline (matches request_payloads.json).")
    p.add_argument("--baseline-batch-size", type=int, default=16,
                   help="Pipeline batch size for GPU saturation.")
    p.add_argument("--baseline-device", type=int, default=0,
                   help="CUDA device index for the baseline pipeline (-1 for CPU).")
    p.add_argument("--no-latency-pass", action="store_true",
                   help="Skip the sequential per-request latency pass.")
    args = p.parse_args()

    if args.baseline:
        run_baseline(
            args.eval,
            args.output,
            threshold=args.baseline_threshold,
            batch_size=args.baseline_batch_size,
            latency_pass=not args.no_latency_pass,
            device=args.baseline_device,
        )
    else:
        run_sweep(
            args.eval,
            args.output,
            model=args.model,
            splitter_backend=args.splitter_backend,
            splitter_language=args.splitter_language,
            device=args.device,
            latency_pass=not args.no_latency_pass,
            encode_batch_size=args.encode_batch_size,
        )


if __name__ == "__main__":
    main()
