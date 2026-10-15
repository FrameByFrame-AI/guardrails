#!/usr/bin/env python3
"""Detailed eval of the multi-label ByteHybrid checkpoint.

Runs the model once on a parquet, caches logits, and then sweeps thresholds
to report:
  - is_valid accuracy / precision / recall vs threshold
  - macro / micro F1 vs threshold
  - JSON exact-match vs threshold
  - per-class F1 at the best threshold
  - error breakdown by `kind` (single / multi / benign)
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
from code_language_id.multilabel_dataset import (
    LANGUAGES,
    NUM_LANGUAGES,
    CodeLanguageMultilabelDataset,
)


def encode_bytes(texts, max_len):
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


def encode_byte_chunks(chunks: list[bytes], max_len: int) -> torch.Tensor:
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


def compute_metrics(probs, labels, is_valid_true, threshold):
    threshold_arr = np.asarray(threshold, dtype=np.float32)
    preds = probs >= threshold_arr
    pred_is_valid = preds.any(axis=1)
    is_valid_acc = float((pred_is_valid == is_valid_true).mean())
    # binary metrics on is_valid
    tp = int(np.logical_and(pred_is_valid, is_valid_true).sum())
    fp = int(np.logical_and(pred_is_valid, ~is_valid_true).sum())
    fn = int(np.logical_and(~pred_is_valid, is_valid_true).sum())
    tn = int(np.logical_and(~pred_is_valid, ~is_valid_true).sum())
    is_valid_p = tp / max(tp + fp, 1)
    is_valid_r = tp / max(tp + fn, 1)

    # per-class
    f1s = []
    per_class = {}
    tp_t = fp_t = fn_t = 0
    for c, lang in enumerate(LANGUAGES):
        tpc = int(np.logical_and(preds[:, c], labels[:, c]).sum())
        fpc = int(np.logical_and(preds[:, c], ~labels[:, c]).sum())
        fnc = int(np.logical_and(~preds[:, c], labels[:, c]).sum())
        tp_t += tpc; fp_t += fpc; fn_t += fnc
        if tpc + fpc + fnc == 0:
            continue
        prec = tpc / max(tpc + fpc, 1)
        rec = tpc / max(tpc + fnc, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        per_class[lang] = (f1, prec, rec, tpc, fpc, fnc)
        f1s.append(f1)
    micro_p = tp_t / max(tp_t + fp_t, 1)
    micro_r = tp_t / max(tp_t + fn_t, 1)
    micro_f1 = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # exact match
    exact = 0
    for i in range(len(preds)):
        if pred_is_valid[i] != is_valid_true[i]:
            continue
        if not is_valid_true[i]:
            exact += 1
            continue
        if np.array_equal(preds[i], labels[i]):
            exact += 1
    exact_match = exact / max(len(preds), 1)

    return {
        "threshold": threshold_arr.tolist() if threshold_arr.ndim else float(threshold_arr),
        "is_valid_acc": is_valid_acc,
        "is_valid_precision": is_valid_p,
        "is_valid_recall": is_valid_r,
        "is_valid_tp": tp, "is_valid_fp": fp, "is_valid_fn": fn, "is_valid_tn": tn,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "json_exact_match": exact_match,
        "per_class": per_class,
    }


def tune_per_class_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    is_valid_true: np.ndarray,
    candidates: list[float],
    start_threshold: float,
    passes: int = 2,
) -> tuple[np.ndarray, dict]:
    """Greedy coordinate search for thresholds that maximize exact JSON match."""
    thresholds = np.full(probs.shape[1], start_threshold, dtype=np.float32)
    best = compute_metrics(probs, labels, is_valid_true, thresholds)
    best_score = (best["json_exact_match"], best["macro_f1"], best["micro_f1"])

    for _ in range(max(1, passes)):
        improved = False
        for c in range(probs.shape[1]):
            class_best_threshold = float(thresholds[c])
            class_best_metrics = best
            class_best_score = best_score
            for candidate in candidates:
                trial = thresholds.copy()
                trial[c] = candidate
                metrics = compute_metrics(probs, labels, is_valid_true, trial)
                score = (
                    metrics["json_exact_match"],
                    metrics["macro_f1"],
                    metrics["micro_f1"],
                )
                if score > class_best_score:
                    class_best_score = score
                    class_best_threshold = candidate
                    class_best_metrics = metrics
            if class_best_threshold != float(thresholds[c]):
                thresholds[c] = class_best_threshold
                best = class_best_metrics
                best_score = class_best_score
                improved = True
        if not improved:
            break
    return thresholds, best


def checkpoint_thresholds(ckpt_path: Path) -> tuple[float, dict[str, float] | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    global_threshold = float(ckpt.get("is_valid_threshold", 0.5))
    per_class_raw = ckpt.get("per_class_thresholds")
    if per_class_raw is None:
        return global_threshold, None
    if isinstance(per_class_raw, dict):
        return global_threshold, {str(k): float(v) for k, v in per_class_raw.items()}
    idx2lang = {int(v): k for k, v in ckpt["lang2idx"].items()}
    return global_threshold, {
        idx2lang[i]: float(v) for i, v in enumerate(per_class_raw)
    }


def write_thresholds_to_checkpoint(
    ckpt_path: Path,
    thresholds: np.ndarray,
    metrics: dict,
    backup: bool = True,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    idx2lang = {int(v): k for k, v in ckpt["lang2idx"].items()}
    per_class = {idx2lang[i]: float(v) for i, v in enumerate(thresholds.tolist())}
    if backup:
        backup_path = ckpt_path.with_suffix(ckpt_path.suffix + ".threshold-backup")
        if not backup_path.exists():
            torch.save(ckpt, backup_path)
            print(f"wrote checkpoint backup: {backup_path}", flush=True)
    ckpt["per_class_thresholds"] = per_class
    ckpt["threshold_tuning"] = {
        "metric": "json_exact_match",
        "json_exact_match": float(metrics["json_exact_match"]),
        "macro_f1": float(metrics["macro_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
    }
    torch.save(ckpt, ckpt_path)
    print(f"updated checkpoint thresholds: {ckpt_path}", flush=True)


@torch.no_grad()
def run_model(
    ckpt_path,
    parquet_path,
    device,
    max_len_override=None,
    batch_size=128,
    window_stride=0,
    window_agg="max",
):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    max_len = max_len_override or ckpt["max_len"]
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=max_len,
        pool_type=ckpt.get("pool_type", "mean"),
        **CONFIGS[ckpt["config"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    ds = CodeLanguageMultilabelDataset(parquet_path)
    if window_stride <= 0:
        probs_list = []
        for i in range(0, len(ds), batch_size):
            rows = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            b = encode_bytes([r["text"] for r in rows], max_len).to(device)
            logits = model(b).float()
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
    else:
        all_chunks: list[bytes] = []
        owners: list[int] = []
        offsets: list[tuple[int, int]] = []
        for i in range(len(ds)):
            rows = ds[i]
            chunks = byte_windows(rows["text"], max_len, window_stride)
            offsets.append((len(all_chunks), len(all_chunks) + len(chunks)))
            all_chunks.extend(chunks)
            owners.extend([i] * len(chunks))

        window_probs_list = []
        for i in range(0, len(all_chunks), batch_size):
            b = encode_byte_chunks(all_chunks[i : i + batch_size], max_len).to(device)
            logits = model(b).float()
            window_probs_list.append(torch.sigmoid(logits).cpu().numpy())
        all_window_probs = np.concatenate(window_probs_list, axis=0)
        probs = np.zeros((len(ds), NUM_LANGUAGES), dtype=np.float32)
        for i, (start, end) in enumerate(offsets):
            probs[i] = aggregate_window_probs(all_window_probs[start:end], window_agg)
        print(
            f"windowed inference: rows={len(ds)} windows={len(all_chunks)} "
            f"stride={window_stride} agg={window_agg}",
            flush=True,
        )
    labels = ds._labels.astype(bool)
    is_valid_true = ds._is_valid.copy()
    kinds = ds.frame["kind"].to_numpy()
    texts = ds.frame["prompt"].to_numpy()
    return probs, labels, is_valid_true, kinds, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(Path(
        "~/llm_models/guardrail_code_models/"
        "programming-language-identification-25-multilabel/model.pt"
    ).expanduser()))
    p.add_argument("--parquet", default=str(Path(
        "~/llm_models/Accuknoxtechnologies-CodeLanguage/data/test-00000-of-00001.parquet"
    ).expanduser()))
    p.add_argument("--device", default=None)
    p.add_argument("--max-len", type=int, default=None,
                   help="Override checkpoint's max_len (e.g. raise to 4096 for longer prompts).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--window-stride", type=int, default=0,
                   help="If >0, run overlapping byte windows with this stride and aggregate probabilities.")
    p.add_argument("--window-agg", choices=["max", "noisy_or"], default="max")
    p.add_argument("--thresholds", default="0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    p.add_argument("--tune-per-class", action="store_true",
                   help="Greedily tune one threshold per language using --thresholds as candidates.")
    p.add_argument("--tune-passes", type=int, default=2)
    p.add_argument("--write-thresholds", type=Path, default=None,
                   help="Write tuned per-class thresholds to this JSON file.")
    p.add_argument("--update-checkpoint-thresholds", action="store_true",
                   help="Persist tuned per-class thresholds into --checkpoint.")
    p.add_argument("--show-errors", type=int, default=10,
                   help="Print N misclassified rows at the best threshold.")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading {args.checkpoint} on {device}", flush=True)
    probs, labels, is_valid_true, kinds, texts = run_model(
        args.checkpoint,
        args.parquet,
        device,
        max_len_override=args.max_len,
        batch_size=args.batch_size,
        window_stride=args.window_stride,
        window_agg=args.window_agg,
    )
    print(f"eval rows: {len(probs)} | kinds: {dict(zip(*np.unique(kinds, return_counts=True)))}")
    print()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    ckpt_global_threshold, ckpt_per_class = checkpoint_thresholds(Path(args.checkpoint))
    results = [compute_metrics(probs, labels, is_valid_true, t) for t in thresholds]

    print(f"{'thr':>5} {'isv_acc':>7} {'isv_P':>6} {'isv_R':>6} {'macF1':>6} {'micF1':>6} {'exact':>6}")
    for r in results:
        print(f"{r['threshold']:>5.2f} {r['is_valid_acc']:>7.4f} "
              f"{r['is_valid_precision']:>6.4f} {r['is_valid_recall']:>6.4f} "
              f"{r['macro_f1']:>6.4f} {r['micro_f1']:>6.4f} {r['json_exact_match']:>6.4f}")

    best = max(results, key=lambda r: r["json_exact_match"])
    if ckpt_per_class is not None:
        vector = np.array([ckpt_per_class[lang] for lang in LANGUAGES], dtype=np.float32)
        saved = compute_metrics(probs, labels, is_valid_true, vector)
        print(f"saved per-class thresholds -> exact={saved['json_exact_match']:.4f} "
              f"macF1={saved['macro_f1']:.4f} micF1={saved['micro_f1']:.4f}")
    if args.tune_per_class:
        tuned_thresholds, tuned = tune_per_class_thresholds(
            probs,
            labels,
            is_valid_true,
            thresholds,
            start_threshold=float(best["threshold"]),
            passes=args.tune_passes,
        )
        best = tuned
        print()
        print("=== tuned per-class thresholds ===")
        for lang, thr in zip(LANGUAGES, tuned_thresholds.tolist()):
            print(f"  {lang:14s} {thr:.3f}")
        print(f"  exact={tuned['json_exact_match']:.4f} "
              f"macF1={tuned['macro_f1']:.4f} micF1={tuned['micro_f1']:.4f}")
        if args.write_thresholds is not None:
            payload = {
                "global_start_threshold": float(best["threshold"]) if np.asarray(best["threshold"]).ndim == 0 else ckpt_global_threshold,
                "per_class_thresholds": {
                    lang: float(thr) for lang, thr in zip(LANGUAGES, tuned_thresholds.tolist())
                },
                "metrics": {
                    "json_exact_match": float(tuned["json_exact_match"]),
                    "macro_f1": float(tuned["macro_f1"]),
                    "micro_f1": float(tuned["micro_f1"]),
                    "is_valid_acc": float(tuned["is_valid_acc"]),
                },
            }
            args.write_thresholds.parent.mkdir(parents=True, exist_ok=True)
            with args.write_thresholds.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            print(f"wrote thresholds: {args.write_thresholds}", flush=True)
        if args.update_checkpoint_thresholds:
            write_thresholds_to_checkpoint(
                Path(args.checkpoint),
                tuned_thresholds,
                tuned,
            )
    print()
    print(f"=== best by json_exact: threshold={best['threshold']} -> exact={best['json_exact_match']:.4f} ===")
    print(f"  is_valid: tp={best['is_valid_tp']} fp={best['is_valid_fp']} "
          f"fn={best['is_valid_fn']} tn={best['is_valid_tn']}")
    print()
    print("  per-class F1 (sorted asc):")
    for lang, (f1, prec, rec, tp, fp, fn) in sorted(best["per_class"].items(), key=lambda x: x[1][0]):
        print(f"    {lang:14s} F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  (tp={tp} fp={fp} fn={fn})")

    # Per-kind error breakdown at best threshold
    preds = probs >= best["threshold"]
    pred_is_valid = preds.any(axis=1)
    print()
    print(f"  error breakdown by `kind` at threshold={best['threshold']}:")
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        wrong_isv = (pred_is_valid[mask] != is_valid_true[mask]).sum()
        if kind == "benign":
            wrong_total = wrong_isv
        else:
            # for valid rows: count rows where either is_valid wrong or category set differs
            wrong_total = 0
            for i in np.where(mask)[0]:
                if pred_is_valid[i] != is_valid_true[i]:
                    wrong_total += 1
                elif not np.array_equal(preds[i], labels[i]):
                    wrong_total += 1
        n = int(mask.sum())
        print(f"    {kind:8s}  total={n}  wrong={wrong_total}  ({wrong_total/n*100:.1f}%)  "
              f"is_valid_wrong={int(wrong_isv)}")

    # Show some error examples
    if args.show_errors > 0:
        print()
        print(f"  sample errors at threshold={best['threshold']}:")
        shown = 0
        for i in range(len(preds)):
            if shown >= args.show_errors:
                break
            wrong = False
            if pred_is_valid[i] != is_valid_true[i]:
                wrong = True
            elif is_valid_true[i] and not np.array_equal(preds[i], labels[i]):
                wrong = True
            if not wrong:
                continue
            shown += 1
            true_cats = sorted(LANGUAGES[j] for j in range(NUM_LANGUAGES) if labels[i, j])
            pred_cats = sorted(LANGUAGES[j] for j in range(NUM_LANGUAGES) if preds[i, j])
            print(f"    [{kinds[i]}] expected is_valid={bool(is_valid_true[i])} cats={true_cats}")
            print(f"             got      is_valid={bool(pred_is_valid[i])} cats={pred_cats}")
            top = sorted(enumerate(probs[i]), key=lambda x: -x[1])[:5]
            print(f"             top probs: " + ", ".join(f"{LANGUAGES[j]}={p:.3f}" for j, p in top))
            preview = texts[i][:120].replace("\n", " ")
            print(f"             prompt[:120]={preview!r}")
            print()


if __name__ == "__main__":
    main()
