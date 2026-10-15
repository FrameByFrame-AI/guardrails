#!/usr/bin/env python3
"""Benchmark guardrail code-language models against philomath-1209.

Compares any of:
  * the full ModernBERT model (``--guardrail-dir``)
  * the lite ByteHybrid model (``--lite-checkpoint``)
  * philomath-1209 (third-party baseline)

The cross-model comparison is restricted to canonical languages that appear in
all enabled models' label spaces. The lite model is also reported on the full
guardrail label set so we can quantify the lite-vs-full quality gap.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.byte_hybrid import CONFIGS, ByteHybrid


GUARDRAIL_MODEL_DIR = Path("/models/guardrail_code_models/modernbert-language-v1")
LITE_MODEL_CHECKPOINT = Path(
    "/models/guardrail_code_models/"
    "programming-language-identification-100plus-lite/model.pt"
)
PHILOMATH_MODEL_ID = "philomath-1209/programming-language-identification"
PHILOMATH_LOCAL_PATH = Path(
    "/models/philomath-programming-language-identification"
)
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
    warmup_batches: int = 0,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    if warmup_batches > 0 and len(texts) >= batch_size:
        warm = texts[:batch_size]
        warm_enc = tokenizer(warm, padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")
        warm_enc = {k: v.to(device) for k, v in warm_enc.items()}
        for _ in range(warmup_batches):
            _ = model(**warm_enc).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
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
    if device.type == "cuda":
        torch.cuda.synchronize()
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
        "warmup_batches": warmup_batches,
    }
    del model
    torch.cuda.empty_cache()
    return predictions, elapsed, meta


@torch.no_grad()
def run_byte_hybrid(
    checkpoint: Path,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
    warmup_batches: int = 0,
) -> tuple[np.ndarray, float, dict[str, Any], dict[int, str]]:
    """Run the lite ByteHybrid checkpoint and return (predictions, elapsed, meta, id2label)."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=ckpt["max_len"],
        **CONFIGS[ckpt["config"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()

    max_len = int(ckpt["max_len"])
    idx2lang = {int(v): k for k, v in ckpt["lang2idx"].items()}

    predictions = np.empty(len(texts), dtype=object)
    if warmup_batches > 0 and len(texts) >= batch_size:
        warm_arr = np.full((batch_size, max_len), 256, dtype=np.int64)
        for i, t in enumerate(texts[:batch_size]):
            raw = t.encode("utf-8", errors="replace")[:max_len]
            if raw:
                warm_arr[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        warm_t = torch.from_numpy(warm_arr).to(device)
        for _ in range(warmup_batches):
            _ = model(warm_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
    start = time.time()
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        encoded = np.full((len(batch), max_len), 256, dtype=np.int64)
        for i, t in enumerate(batch):
            raw = t.encode("utf-8", errors="replace")[:max_len]
            if raw:
                encoded[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        byte_ids = torch.from_numpy(encoded).to(device)
        logits = model(byte_ids).float()
        ids = logits.argmax(dim=-1).detach().cpu().tolist()
        for offset, label_id in enumerate(ids):
            predictions[batch_start + offset] = idx2lang[int(label_id)]
        if batch_start // batch_size % 20 == 0:
            print(
                f"  [byte-hybrid lite] {batch_start + len(batch)}/{len(texts)}"
                f" elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    meta = {
        "model_id": str(checkpoint),
        "num_labels": int(ckpt["num_classes"]),
        "batch_size": batch_size,
        "max_len": max_len,
        "config": ckpt["config"],
        "dtype": str(dtype).split(".")[-1],
        "device": str(device),
        "elapsed_seconds": elapsed,
        "warmup_batches": warmup_batches,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, elapsed, meta, idx2lang


def run_onnx_seq_classifier(
    onnx_path: Path,
    tokenizer_dir: Path,
    texts: list[str],
    id2label: dict[int, str],
    batch_size: int,
    max_length: int,
    providers: list[str],
    warmup_batches: int = 0,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run a HF-style sequence classifier exported to ONNX (RoBERTa, BERT, etc.)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    expected_inputs = {i.name for i in sess.get_inputs()}

    def encode(batch_texts: list[str]) -> dict[str, np.ndarray]:
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed: dict[str, np.ndarray] = {}
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            if name in encoded and name in expected_inputs:
                feed[name] = encoded[name].astype(np.int64)
        return feed

    if warmup_batches > 0 and len(texts) >= 1:
        warm_feed = encode(texts[:batch_size])
        for _ in range(warmup_batches):
            sess.run(None, warm_feed)

    predictions = np.empty(len(texts), dtype=object)
    start = time.time()
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        feed = encode(batch)
        logits = sess.run(None, feed)[0]
        for offset, label_id in enumerate(logits.argmax(axis=-1).tolist()):
            predictions[batch_start + offset] = id2label[int(label_id)]
        if batch_start // batch_size % 20 == 0:
            print(
                f"  [onnx {onnx_path.name}] {batch_start + len(batch)}/{len(texts)}"
                f" elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    elapsed = time.time() - start
    meta = {
        "model_id": str(onnx_path),
        "tokenizer_id": str(tokenizer_dir),
        "num_labels": len(id2label),
        "batch_size": batch_size,
        "max_length": max_length,
        "providers": providers,
        "elapsed_seconds": elapsed,
        "warmup_batches": warmup_batches,
        "runtime": "onnxruntime",
    }
    return predictions, elapsed, meta


def run_byte_hybrid_onnx(
    onnx_path: Path,
    lang2idx: dict[str, int],
    max_len: int,
    texts: list[str],
    batch_size: int,
    providers: list[str],
    warmup_batches: int = 0,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run the lite ByteHybrid model exported to ONNX."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    idx2lang = {int(v): k for k, v in lang2idx.items()}

    def encode(batch_texts: list[str]) -> np.ndarray:
        out = np.full((len(batch_texts), max_len), 256, dtype=np.int64)
        for i, t in enumerate(batch_texts):
            raw = t.encode("utf-8", errors="replace")[:max_len]
            if raw:
                out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        return out

    if warmup_batches > 0 and len(texts) >= 1:
        warm = encode(texts[:batch_size])
        for _ in range(warmup_batches):
            sess.run(None, {"byte_ids": warm})

    predictions = np.empty(len(texts), dtype=object)
    start = time.time()
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        byte_ids = encode(batch)
        logits = sess.run(None, {"byte_ids": byte_ids})[0]
        for offset, label_id in enumerate(logits.argmax(axis=-1).tolist()):
            predictions[batch_start + offset] = idx2lang[int(label_id)]
        if batch_start // batch_size % 20 == 0:
            print(
                f"  [onnx byte-hybrid] {batch_start + len(batch)}/{len(texts)}"
                f" elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    elapsed = time.time() - start
    meta = {
        "model_id": str(onnx_path),
        "num_labels": len(idx2lang),
        "batch_size": batch_size,
        "max_len": max_len,
        "providers": providers,
        "elapsed_seconds": elapsed,
        "warmup_batches": warmup_batches,
        "runtime": "onnxruntime",
    }
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
    parser.add_argument(
        "--lite-checkpoint",
        type=Path,
        default=None,
        help="Path to a ByteHybrid lite model.pt. If omitted, the lite model is skipped.",
    )
    parser.add_argument(
        "--lite-onnx",
        type=Path,
        default=None,
        help="Path to a ByteHybrid lite ONNX model. Adds an onnxruntime row to the comparison.",
    )
    parser.add_argument(
        "--philomath-onnx",
        type=Path,
        default=None,
        help="Path to a philomath ONNX model file (e.g. /models/philomath-.../onnx/model.onnx). "
             "Tokenizer is loaded from --philomath-path.",
    )
    parser.add_argument(
        "--onnx-providers",
        default="CPUExecutionProvider",
        help="Comma-separated onnxruntime providers, e.g. 'CUDAExecutionProvider,CPUExecutionProvider' "
             "or 'CPUExecutionProvider' for CPU-only timings.",
    )
    parser.add_argument(
        "--skip-guardrail",
        action="store_true",
        help="Skip the full ModernBERT guardrail model.",
    )
    parser.add_argument(
        "--skip-philomath",
        action="store_true",
        help="Skip the philomath-1209 baseline.",
    )
    parser.add_argument(
        "--philomath-path",
        type=Path,
        default=None,
        help="Local directory of the philomath model (e.g. /models/philomath-programming-language-identification). "
             "If a directory exists at this path, it is loaded directly; otherwise the script falls back "
             "to --philomath-id from the HF hub.",
    )
    parser.add_argument("--philomath-id", default=PHILOMATH_MODEL_ID)
    parser.add_argument(
        "--device",
        default=None,
        help="Force a torch device (cpu / cuda / cuda:0). Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Warm-up batches discarded from the throughput timer (per model).",
    )
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

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    guardrail_attn = "eager"
    philomath_attn = None

    philomath_source: str
    if args.philomath_path and Path(args.philomath_path).exists():
        philomath_source = str(args.philomath_path)
        philomath_tokenizer_id: str | None = str(args.philomath_path)
    else:
        philomath_source = args.philomath_id
        philomath_tokenizer_id = None

    frame = load_split(args.splits_root, args.split)
    print(f"Loaded {args.split} split: {len(frame)} rows", flush=True)

    runs: dict[str, dict[str, Any]] = {}

    guardrail_id2label: dict[int, str] | None = None
    if not args.skip_guardrail:
        guardrail_id2label = load_guardrail_id2label(args.guardrail_dir)
        runs["guardrail"] = {
            "labels": set(guardrail_id2label.values()),
            "label_count": len(guardrail_id2label),
        }
    lite_lang2idx: dict[str, int] | None = None
    lite_max_len: int | None = None
    if args.lite_checkpoint is not None or args.lite_onnx is not None:
        # Both torch and onnx variants share lang2idx/max_len. We need a checkpoint
        # to know them when only --lite-onnx is given.
        meta_source = args.lite_checkpoint
        if meta_source is None:
            meta_source = args.lite_onnx.parent.parent / "model.pt"
        ckpt_meta = torch.load(str(meta_source), map_location="cpu", weights_only=False)
        lite_lang2idx = dict(ckpt_meta["lang2idx"])
        lite_max_len = int(ckpt_meta["max_len"])
        lite_labels = set(lite_lang2idx.keys())
        del ckpt_meta
    if args.lite_checkpoint is not None:
        runs["lite"] = {"labels": lite_labels, "label_count": len(lite_labels)}
    if args.lite_onnx is not None:
        runs["lite_onnx"] = {"labels": lite_labels, "label_count": len(lite_labels)}
    philomath_id2label: dict[int, str] | None = None
    if not args.skip_philomath:
        philomath_id2label = load_philomath_id2label()
        runs["philomath"] = {
            "labels": set(philomath_id2label.values()),
            "label_count": len(philomath_id2label),
        }
    if args.philomath_onnx is not None and not args.skip_philomath:
        runs["philomath_onnx"] = {
            "labels": set(philomath_id2label.values()),
            "label_count": len(philomath_id2label),
        }

    if not runs:
        raise SystemExit("No models enabled — nothing to do.")

    shared_labels = sorted(set.intersection(*[r["labels"] for r in runs.values()]))
    print(
        " | ".join(f"{name}: {info['label_count']}" for name, info in runs.items())
        + f" | shared: {len(shared_labels)}",
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

    if "guardrail" in runs:
        print(f"\n== Running guardrail ModernBERT on {device} ==", flush=True)
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
            warmup_batches=args.warmup_batches,
        )
        runs["guardrail"]["pred"] = guardrail_pred
        runs["guardrail"]["meta"] = guardrail_meta
        runs["guardrail"]["display_name"] = "guardrail-modernbert"

    if "lite" in runs:
        print(f"\n== Running lite ByteHybrid {args.lite_checkpoint} on {device} ==", flush=True)
        lite_pred, _, lite_meta, _ = run_byte_hybrid(
            checkpoint=args.lite_checkpoint,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            dtype=dtype,
            warmup_batches=args.warmup_batches,
        )
        runs["lite"]["pred"] = lite_pred
        runs["lite"]["meta"] = lite_meta
        runs["lite"]["display_name"] = "guardrail-lite-bytehybrid"

    if "philomath" in runs:
        print(f"\n== Running {philomath_source} on {device} ==", flush=True)
        philomath_pred, _, philomath_meta = run_model(
            model_id=philomath_source,
            tokenizer_id=philomath_tokenizer_id,
            texts=texts,
            id2label=philomath_id2label,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            attn_implementation=philomath_attn,
            dtype=dtype,
            warmup_batches=args.warmup_batches,
        )
        runs["philomath"]["pred"] = philomath_pred
        runs["philomath"]["meta"] = philomath_meta
        runs["philomath"]["display_name"] = "philomath-1209"

    onnx_providers = [p.strip() for p in args.onnx_providers.split(",") if p.strip()]

    if "lite_onnx" in runs:
        print(f"\n== Running lite ByteHybrid ONNX {args.lite_onnx} ({onnx_providers}) ==", flush=True)
        lite_onnx_pred, _, lite_onnx_meta = run_byte_hybrid_onnx(
            onnx_path=args.lite_onnx,
            lang2idx=lite_lang2idx,
            max_len=lite_max_len,
            texts=texts,
            batch_size=args.batch_size,
            providers=onnx_providers,
            warmup_batches=args.warmup_batches,
        )
        runs["lite_onnx"]["pred"] = lite_onnx_pred
        runs["lite_onnx"]["meta"] = lite_onnx_meta
        runs["lite_onnx"]["display_name"] = "lite-bytehybrid-onnx"

    if "philomath_onnx" in runs:
        tokenizer_dir = Path(args.philomath_path) if args.philomath_path else Path(philomath_source)
        print(f"\n== Running philomath ONNX {args.philomath_onnx} ({onnx_providers}) ==", flush=True)
        philo_onnx_pred, _, philo_onnx_meta = run_onnx_seq_classifier(
            onnx_path=args.philomath_onnx,
            tokenizer_dir=tokenizer_dir,
            texts=texts,
            id2label=philomath_id2label,
            batch_size=args.batch_size,
            max_length=args.max_length,
            providers=onnx_providers,
            warmup_batches=args.warmup_batches,
        )
        runs["philomath_onnx"]["pred"] = philo_onnx_pred
        runs["philomath_onnx"]["meta"] = philo_onnx_meta
        runs["philomath_onnx"]["display_name"] = "philomath-1209-onnx"

    for name, info in runs.items():
        info["summary"] = summarize(gold, info["pred"], shared_labels)

    print("\n== Summary (restricted to shared labels) ==")
    fmt = "{:<30s} {:>10s} {:>10s} {:>10s} {:>12s} {:>10s}"
    print(fmt.format("model", "accuracy", "macro_f1", "weighted_f1", "elapsed_s", "texts/s"))
    n_total = int(len(texts))
    for name, info in runs.items():
        s = info["summary"]
        elapsed = float(info["meta"].get("elapsed_seconds", 0.0))
        tps = n_total / elapsed if elapsed > 0 else 0.0
        info["meta"]["texts_per_second"] = tps
        print(
            fmt.format(
                info["display_name"],
                f"{s['accuracy']:.4f}",
                f"{s['macro_f1']:.4f}",
                f"{s['weighted_f1']:.4f}",
                f"{elapsed:.2f}",
                f"{tps:.1f}",
            )
        )

    print("\nPer-language F1 (shared labels):")
    header = ["language", "support"] + [info["display_name"][:14] for info in runs.values()]
    print(("{:<25s}" + "{:>14s}" * (len(header) - 1)).format(*header))
    for label in shared_labels:
        row: list[Any] = [label]
        first = next(iter(runs.values()))
        row.append(first["summary"]["per_label"][label]["support"])
        for info in runs.values():
            row.append(info["summary"]["per_label"][label]["f1"])
        line = "{:<25s}{:>14d}".format(row[0], row[1]) + "".join(
            "{:>14.4f}".format(v) for v in row[2:]
        )
        print(line)

    payload: dict[str, Any] = {
        "split": args.split,
        "shared_labels": shared_labels,
        "num_restricted_rows": int(len(restricted)),
        "num_original_rows": int(len(frame)),
        "max_chars": args.max_chars,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "models": {
            name: {"summary": info["summary"], "meta": info["meta"]}
            for name, info in runs.items()
        },
    }

    if "lite" in runs and "guardrail" in runs:
        full_labels = sorted(runs["guardrail"]["labels"])
        lite_full_pred = runs["lite"]["pred"]
        guardrail_full_pred = runs["guardrail"]["pred"]
        full_frame = frame[frame["canonical_language"].isin(full_labels)].reset_index(drop=True)
        if args.limit is not None:
            full_frame = full_frame.head(args.limit).reset_index(drop=True)
        if len(full_frame) != len(restricted) or set(full_labels) != set(shared_labels):
            print(
                "\n[note] full-label lite-vs-full comparison requires running both models on the"
                " full guardrail label set; rerun with --skip-philomath to get an apples-to-apples"
                " 100plus-label table.",
                flush=True,
            )
        else:
            full_summary_lite = summarize(gold, lite_full_pred, full_labels)
            full_summary_guard = summarize(gold, guardrail_full_pred, full_labels)
            payload["full_label_comparison"] = {
                "labels": full_labels,
                "lite": full_summary_lite,
                "guardrail": full_summary_guard,
            }
            print("\n== Full guardrail label set ({} labels) ==".format(len(full_labels)))
            print(fmt.format("model", "accuracy", "macro_f1", "weighted_f1"))
            print(
                fmt.format(
                    "guardrail-modernbert",
                    f"{full_summary_guard['accuracy']:.4f}",
                    f"{full_summary_guard['macro_f1']:.4f}",
                    f"{full_summary_guard['weighted_f1']:.4f}",
                )
            )
            print(
                fmt.format(
                    "lite-bytehybrid",
                    f"{full_summary_lite['accuracy']:.4f}",
                    f"{full_summary_lite['macro_f1']:.4f}",
                    f"{full_summary_lite['weighted_f1']:.4f}",
                )
            )

    output_path = args.output_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
