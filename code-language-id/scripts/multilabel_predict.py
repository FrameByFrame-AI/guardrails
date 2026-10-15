#!/usr/bin/env python3
"""Predict with the multi-label ByteHybrid checkpoint.

Output format matches the Accuknox CodeLanguage dataset's `target` schema:

    {"is_valid": bool, "category": {"<Lang>": true, ...}}

`is_valid` is false (and `category` empty) when no per-language sigmoid prob
exceeds the threshold; otherwise true with all languages above threshold.

    python scripts/multilabel_predict.py "def hello(): pass"
    python scripts/multilabel_predict.py --file snippet.go
    cat prompts.txt | python scripts/multilabel_predict.py --stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.byte_hybrid import CONFIGS, ByteHybrid


def load(checkpoint: str | Path, dtype: str = "fp32", device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=ckpt["max_len"],
        pool_type=ckpt.get("pool_type", "mean"),
        **CONFIGS[ckpt["config"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval().to(device_t)
    idx2lang = {v: k for k, v in ckpt["lang2idx"].items()}
    default_threshold = float(ckpt.get("is_valid_threshold", 0.5))
    per_class_raw = ckpt.get("per_class_thresholds")
    per_class_thresholds = None
    if per_class_raw is not None:
        if isinstance(per_class_raw, dict):
            per_class_thresholds = np.array(
                [float(per_class_raw[idx2lang[i]]) for i in range(len(idx2lang))],
                dtype=np.float32,
            )
        else:
            per_class_thresholds = np.array(per_class_raw, dtype=np.float32)
    return model, idx2lang, ckpt["max_len"], device_t, default_threshold, per_class_thresholds


def encode(texts: list[str], max_len: int) -> torch.Tensor:
    out = np.full((len(texts), max_len), 256, dtype=np.int64)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        raw = t.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


def byte_windows(text: str, max_len: int, stride: int) -> list[bytes]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_len or stride <= 0:
        return [raw[:max_len]]
    windows: list[bytes] = []
    for start in range(0, len(raw), stride):
        chunk = raw[start : start + max_len]
        if not chunk:
            break
        windows.append(chunk)
        if start + max_len >= len(raw):
            break
    return windows


def encode_chunks(chunks: list[bytes], max_len: int) -> torch.Tensor:
    out = np.full((len(chunks), max_len), 256, dtype=np.int64)
    for i, raw in enumerate(chunks):
        raw = raw[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


def aggregate_window_probs(window_probs: np.ndarray, method: str) -> np.ndarray:
    if len(window_probs) == 1:
        return window_probs[0]
    if method == "max":
        return window_probs.max(axis=0)
    if method == "noisy_or":
        return 1.0 - np.prod(1.0 - window_probs, axis=0)
    raise ValueError(f"unknown window aggregation method: {method!r}")


@torch.no_grad()
def predict(
    model: ByteHybrid,
    texts: list[str],
    idx2lang: dict[int, str],
    max_len: int,
    device: torch.device,
    threshold: float | np.ndarray,
    batch_size: int = 128,
    window_stride: int = 0,
    window_agg: str = "max",
) -> list[dict]:
    out: list[dict] = []
    if window_stride <= 0:
        probs_rows: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            b = encode(texts[i : i + batch_size], max_len).to(device)
            probs = torch.sigmoid(model(b).float()).cpu().numpy()
            probs_rows.extend([row for row in probs])
    else:
        probs_rows = []
        for text in texts:
            chunks = byte_windows(text, max_len, window_stride)
            chunk_probs = []
            for i in range(0, len(chunks), batch_size):
                b = encode_chunks(chunks[i : i + batch_size], max_len).to(device)
                probs = torch.sigmoid(model(b).float()).cpu().numpy()
                chunk_probs.append(probs)
            probs_rows.append(aggregate_window_probs(np.concatenate(chunk_probs, axis=0), window_agg))
    threshold_arr = np.asarray(threshold, dtype=np.float32)
    for row in probs_rows:
        above = row >= threshold_arr
        if not above.any():
            out.append({"is_valid": False, "category": {}})
        else:
            cats = {idx2lang[int(j)]: True for j in np.where(above)[0]}
            out.append({"is_valid": True, "category": cats})
    return out


def _main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="*", help="Text(s) to classify (one per arg).")
    p.add_argument("--file", help="Read a single text from FILE.")
    p.add_argument("--stdin", action="store_true", help="One text per line from stdin.")
    p.add_argument(
        "--checkpoint",
        default=str(
            Path(
                "~/llm_models/guardrail_code_models/"
                "programming-language-identification-25-multilabel/model.pt"
            ).expanduser()
        ),
    )
    p.add_argument("--threshold", type=float, default=None,
                   help="Override sigmoid threshold (default: value baked into the checkpoint).")
    p.add_argument("--ignore-per-class-thresholds", action="store_true",
                   help="Use the scalar threshold even if the checkpoint has per-class thresholds.")
    p.add_argument("--window-stride", type=int, default=0,
                   help="If >0, run overlapping byte windows with this stride and aggregate probabilities.")
    p.add_argument("--window-agg", choices=["max", "noisy_or"], default="max")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--with-probs", action="store_true",
                   help="Include per-language probabilities in the output JSON.")
    args = p.parse_args()

    if args.stdin:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        texts = [Path(args.file).read_text(encoding="utf-8")]
    elif args.text:
        texts = args.text
    else:
        p.print_help()
        return

    model, idx2lang, max_len, device, default_threshold, per_class_thresholds = load(
        args.checkpoint, args.dtype, args.device
    )
    if args.threshold is not None:
        threshold = args.threshold
        threshold_desc = f"{threshold:.3f}"
    elif per_class_thresholds is not None and not args.ignore_per_class_thresholds:
        threshold = per_class_thresholds
        threshold_desc = "per-class"
    else:
        threshold = default_threshold
        threshold_desc = f"{threshold:.3f}"
    print(
        f"# {len(idx2lang)} languages, max_len={max_len}, dtype={args.dtype}, "
        f"device={device}, threshold={threshold_desc}, "
        f"window_stride={args.window_stride}, window_agg={args.window_agg}",
        file=sys.stderr,
    )

    results = predict(
        model,
        texts,
        idx2lang,
        max_len,
        device,
        threshold,
        args.batch_size,
        args.window_stride,
        args.window_agg,
    )

    if args.with_probs:
        # Re-run to grab probs (cheap; same data).
        with torch.no_grad():
            if args.window_stride > 0:
                probs = []
                for text in texts:
                    chunks = byte_windows(text, max_len, args.window_stride)
                    b = encode_chunks(chunks, max_len).to(device)
                    row_probs = torch.sigmoid(model(b).float()).cpu().numpy()
                    probs.append(aggregate_window_probs(row_probs, args.window_agg))
                probs = np.stack(probs, axis=0)
            else:
                b = encode(texts, max_len).to(device)
                probs = torch.sigmoid(model(b).float()).cpu().numpy()
        for text, result, row in zip(texts, results, probs):
            result["_probs"] = {idx2lang[i]: float(row[i]) for i in range(len(idx2lang))}
            print(json.dumps(result, ensure_ascii=False))
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    _main()
