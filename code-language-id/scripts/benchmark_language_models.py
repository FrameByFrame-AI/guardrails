#!/usr/bin/env python3
"""Benchmark modernbert-language-v1 against philomath-1209 on the v1 test split.

The comparison is restricted to the 24 canonical languages that appear in both
models' label spaces. For each model we take the first ``max_chars`` characters
of the code sample (matching the training-time eval strategy) and compare the
argmax prediction against the gold label.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


GUARDRAIL_MODEL_DIR = Path("/models/guardrail_code_models/modernbert-language-v1")
PHILOMATH_MODEL_ID = "philomath-1209/programming-language-identification"
DEFAULT_SPLITS_ROOT = Path("/models/guardrail_code_data/processed/v1_splits")
DEFAULT_LABELS_CSV = Path(
    "/models/guardrail_code_data/interim/reports/v1_languages.csv"
)
DEFAULT_OUTPUT_DIR = Path("/workspace/data/benchmark_results")


def normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n")


def head_snippet(code: str, max_chars: int) -> str:
    code = normalize_code(code)
    return code if len(code) <= max_chars else code[:max_chars]


def load_split(splits_root: Path, split: str) -> pd.DataFrame:
    path = splits_root / f"{split}.parquet"
    frame = pd.read_parquet(path)
    required = {"id", "canonical_language", "code"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def restrict_to_shared(
    frame: pd.DataFrame, shared_labels: list[str]
) -> pd.DataFrame:
    mask = frame["canonical_language"].isin(shared_labels)
    return frame.loc[mask].reset_index(drop=True)


def compute_per_label_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = int(((pred == label) & (gold == label)).sum())
        fp = int(((pred == label) & (gold != label)).sum())
        fn = int(((pred != label) & (gold == label)).sum())
        support = int((gold == label).sum())
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return results


def summarize(
    gold: np.ndarray,
    pred: np.ndarray,
    shared_labels: list[str],
) -> dict[str, Any]:
    per_label = compute_per_label_metrics(gold, pred, shared_labels)
    supports = [per_label[label]["support"] for label in shared_labels]
    f1s = [per_label[label]["f1"] for label in shared_labels]
    precisions = [per_label[label]["precision"] for label in shared_labels]
    recalls = [per_label[label]["recall"] for label in shared_labels]
    weighted_f1 = (
        float(np.average(f1s, weights=supports)) if sum(supports) > 0 else 0.0
    )
    return {
        "accuracy": float((pred == gold).mean()),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": weighted_f1,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "num_samples": int(len(gold)),
        "per_label": per_label,
    }


@torch.no_grad()
def run_model(
    model_id: str,
    tokenizer_id: str | None,
    texts: list[str],
    id2label: dict[int, str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    attn_implementation: str | None,
    dtype: torch.dtype,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
    load_kwargs: dict[str, Any] = {}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, **load_kwargs
    )
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    model.to(device=device, dtype=dtype)
    model.eval()

    predictions = np.empty(len(texts), dtype=object)
    start = time.time()
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for offset, label_id in enumerate(ids):
            predictions[batch_start + offset] = id2label[int(label_id)]
        if batch_start // batch_size % 20 == 0:
            print(
                f"  [{model_id}] {batch_start + len(batch)}/{len(texts)}"
                f" elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    elapsed = time.time() - start
    meta = {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id or model_id,
        "num_labels": len(id2label),
        "batch_size": batch_size,
        "max_length": max_length,
        "attn_implementation": attn_implementation,
        "dtype": str(dtype).split(".")[-1],
        "device": str(device),
        "elapsed_seconds": elapsed,
    }
    del model
    torch.cuda.empty_cache()
    return predictions, elapsed, meta


def load_guardrail_id2label(model_dir: Path) -> dict[int, str]:
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    raw = metadata["id2label"]
    return {int(k): v for k, v in raw.items()}


def load_philomath_id2label() -> dict[int, str]:
    return {
        0: "Scala",
        1: "JavaScript",
        2: "COBOL",
        3: "ARM Assembly",
        4: "R",
        5: "Lua",
        6: "C++",
        7: "Visual Basic .NET",
        8: "Go",
        9: "Erlang",
        10: "C#",
        11: "Rust",
        12: "Ruby",
        13: "Swift",
        14: "Mathematica/Wolfram Language",
        15: "PHP",
        16: "Fortran",
        17: "AppleScript",
        18: "Pascal",
        19: "Java",
        20: "PowerShell",
        21: "Python",
        22: "C",
        23: "Perl",
        24: "Kotlin",
        25: "jq",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--guardrail-dir", type=Path, default=GUARDRAIL_MODEL_DIR
    )
    parser.add_argument("--philomath-id", default=PHILOMATH_MODEL_ID)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--output-name",
        default="language_id_guardrail_vs_philomath.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    guardrail_attn = "eager"
    philomath_attn = None

    frame = load_split(args.splits_root, args.split)
    print(f"Loaded {args.split} split: {len(frame)} rows", flush=True)

    guardrail_id2label = load_guardrail_id2label(args.guardrail_dir)
    philomath_id2label = load_philomath_id2label()

    guardrail_labels = set(guardrail_id2label.values())
    philomath_labels = set(philomath_id2label.values())
    shared_labels = sorted(guardrail_labels & philomath_labels)
    print(
        f"Guardrail labels: {len(guardrail_labels)} |"
        f" Philomath labels: {len(philomath_labels)} |"
        f" Shared: {len(shared_labels)}",
        flush=True,
    )
    print(f"Shared labels: {shared_labels}", flush=True)

    restricted = restrict_to_shared(frame, shared_labels)
    if args.limit is not None:
        restricted = restricted.head(args.limit).reset_index(drop=True)
    print(
        f"Restricted eval rows: {len(restricted)}"
        f" (dropped {len(frame) - len(restricted)} out-of-scope rows)",
        flush=True,
    )

    texts = [head_snippet(code, args.max_chars) for code in restricted["code"]]
    gold = restricted["canonical_language"].to_numpy()

    print(f"\n== Running guardrail modernbert-language-v1 on {device} ==", flush=True)
    guardrail_pred, _, guardrail_meta = run_model(
        model_id=str(args.guardrail_dir),
        tokenizer_id=str(args.guardrail_dir),
        texts=texts,
        id2label=guardrail_id2label,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        attn_implementation=guardrail_attn,
        dtype=dtype,
    )

    print(f"\n== Running {args.philomath_id} on {device} ==", flush=True)
    philomath_pred, _, philomath_meta = run_model(
        model_id=args.philomath_id,
        tokenizer_id=None,
        texts=texts,
        id2label=philomath_id2label,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        attn_implementation=philomath_attn,
        dtype=dtype,
    )

    guardrail_summary = summarize(gold, guardrail_pred, shared_labels)
    philomath_summary = summarize(gold, philomath_pred, shared_labels)

    print("\n== Summary (restricted to shared labels) ==")
    fmt = "{:<30s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("model", "accuracy", "macro_f1", "weighted_f1"))
    print(
        fmt.format(
            "guardrail-modernbert-v1",
            f"{guardrail_summary['accuracy']:.4f}",
            f"{guardrail_summary['macro_f1']:.4f}",
            f"{guardrail_summary['weighted_f1']:.4f}",
        )
    )
    print(
        fmt.format(
            "philomath-1209",
            f"{philomath_summary['accuracy']:.4f}",
            f"{philomath_summary['macro_f1']:.4f}",
            f"{philomath_summary['weighted_f1']:.4f}",
        )
    )

    print("\nPer-language F1 (shared labels):")
    print(
        "{:<25s} {:>8s} {:>10s} {:>10s} {:>6s}".format(
            "language", "support", "guardrail", "philomath", "delta"
        )
    )
    for label in shared_labels:
        g = guardrail_summary["per_label"][label]
        p = philomath_summary["per_label"][label]
        print(
            "{:<25s} {:>8d} {:>10.4f} {:>10.4f} {:>+6.3f}".format(
                label,
                g["support"],
                g["f1"],
                p["f1"],
                g["f1"] - p["f1"],
            )
        )

    payload = {
        "split": args.split,
        "shared_labels": shared_labels,
        "num_restricted_rows": int(len(restricted)),
        "num_original_rows": int(len(frame)),
        "max_chars": args.max_chars,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "guardrail": {
            "summary": guardrail_summary,
            "meta": guardrail_meta,
        },
        "philomath": {
            "summary": philomath_summary,
            "meta": philomath_meta,
        },
    }

    output_path = args.output_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
