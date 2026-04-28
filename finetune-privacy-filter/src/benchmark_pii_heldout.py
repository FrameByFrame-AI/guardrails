#!/usr/bin/env python3
"""
Benchmark Privacy Filter checkpoints on the Korean OPF span test split.

This benchmark intentionally reports only span metrics:
  - exact span match on (label, start, end)
  - micro precision / recall / F1
  - per-label precision / recall / F1

When multiple models are passed, the benchmark labels default to the
intersection of:
  - labels present in the dataset label space
  - labels supported by every compared model

This makes base-vs-finetuned comparisons fair.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Checkpoint path. Legacy form; pass multiple times to compare multiple models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Checkpoint paths as a list.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional display name for each --model, in the same order.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=[],
        help="Display names as a list, matching --models order.",
    )
    parser.add_argument(
        "--dataset",
        default="/data/generated/ko_pii_opf_v2/test.jsonl",
        help="Span-format OPF test dataset.",
    )
    parser.add_argument(
        "--label-space-json",
        default="/data/generated/ko_pii_opf_v2/label_space.json",
        help="Label space JSON for the OPF dataset.",
    )
    parser.add_argument(
        "--output",
        default="/data/benchmark_results/privacy_filter_span_benchmark.json",
    )
    parser.add_argument(
        "--labels-mode",
        choices=["shared", "dataset"],
        default="shared",
        help="shared: compare only labels supported by every model. dataset: compare against all dataset labels.",
    )
    parser.add_argument("--samples", type=int, default=0, help="0 means all test records.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--onnx", action="store_true",
                        help="Load models via optimum.onnxruntime (ONNX format).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Inference device. 'auto' uses CUDA if available.")
    parser.add_argument("--measure-speed", action="store_true", default=True,
                        help="Record per-record inference latency.")
    args = parser.parse_args()

    if args.model and args.models:
        raise ValueError("Use either --model or --models, not both.")
    if args.model_name and args.model_names:
        raise ValueError("Use either --model-name or --model-names, not both.")

    args.model_paths = list(args.models or args.model)
    args.resolved_model_names = list(args.model_names or args.model_name)

    if not args.model_paths:
        raise ValueError("At least one model must be provided via --model or --models.")

    return args


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _sample_records(
    records: list[dict[str, Any]], *, max_samples: int, seed: int
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    sampled = rng.sample(records, max_samples)
    sampled.sort(key=lambda item: str(item.get("info", {}).get("source_id", "")))
    return sampled


def _load_dataset_labels(path: str | Path) -> list[str]:
    payload = _load_json(path)
    span_class_names = payload.get("span_class_names")
    if not isinstance(span_class_names, list) or not span_class_names:
        raise ValueError("label_space.json missing non-empty span_class_names")
    if span_class_names[0] != "O":
        raise ValueError("label_space.json must start span_class_names with 'O'")
    return [str(name) for name in span_class_names if str(name) != "O"]


def _normalize_id2label(mapping: dict[object, object]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_key, raw_value in mapping.items():
        normalized[int(raw_key)] = str(raw_value)
    return normalized


def _extract_categories_from_id2label(id_to_label: dict[int, str]) -> set[str]:
    categories = set()
    for token_label in id_to_label.values():
        if token_label == "O":
            continue
        _, category = token_label.split("-", 1)
        categories.add(category)
    return categories


def _load_model_categories(model_path: str) -> set[str]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    id_to_label = _normalize_id2label(config.id2label)
    return _extract_categories_from_id2label(id_to_label)


def _resolve_model_names(model_paths: list[str], requested_names: list[str]) -> list[str]:
    if requested_names and len(requested_names) != len(model_paths):
        raise ValueError("--model-name count must match --model count")
    if requested_names:
        return requested_names
    resolved = []
    for model_path in model_paths:
        name = Path(model_path.rstrip("/")).name
        resolved.append(name or model_path)
    return resolved


def _extract_gt_spans(
    record: dict[str, Any],
    *,
    allowed_labels: set[str],
) -> set[tuple[str, int, int]]:
    gt_set: set[tuple[str, int, int]] = set()

    label_items = record.get("label")
    if isinstance(label_items, list):
        for item in label_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("category", ""))
            if label not in allowed_labels:
                continue
            gt_set.add((label, int(item["start"]), int(item["end"])))
        return gt_set

    spans_field = record.get("spans")
    if isinstance(spans_field, dict):
        for raw_label, offsets in spans_field.items():
            label = str(raw_label).split(": ", 1)[0]
            if label not in allowed_labels:
                continue
            for start, end in offsets:
                gt_set.add((label, int(start), int(end)))
    return gt_set


def _is_ignorable_boundary_char(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def _trim_span_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    trimmed_start = start
    trimmed_end = end

    while trimmed_start < trimmed_end and _is_ignorable_boundary_char(text[trimmed_start]):
        trimmed_start += 1
    while trimmed_end > trimmed_start and _is_ignorable_boundary_char(text[trimmed_end - 1]):
        trimmed_end -= 1

    if trimmed_start >= trimmed_end:
        return start, end
    return trimmed_start, trimmed_end


def _normalize_span_set(
    text: str, spans: set[tuple[str, int, int]]
) -> set[tuple[str, int, int]]:
    normalized: set[tuple[str, int, int]] = set()
    for label, start, end in spans:
        norm_start, norm_end = _trim_span_boundaries(text, start, end)
        normalized.add((label, norm_start, norm_end))
    return normalized


def _decode_bioes_spans(
    label_ids: list[int], id_to_label: dict[int, str]
) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    active_label: str | None = None
    active_start: int | None = None

    for token_idx, label_id in enumerate(label_ids):
        label_name = id_to_label[int(label_id)]
        if label_name == "O":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
                active_label = None
                active_start = None
            continue

        prefix, category = label_name.split("-", 1)
        if prefix == "S":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
            spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
            continue
        if prefix == "B":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
            active_label = category
            active_start = token_idx
            continue
        if prefix == "I":
            if active_label != category or active_start is None:
                active_label = category
                active_start = token_idx
            continue
        if prefix == "E":
            if active_label == category and active_start is not None:
                spans.add((category, active_start, token_idx + 1))
            else:
                spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
            continue
        raise ValueError(f"Unsupported BIOES label {label_name!r}")

    if active_label is not None and active_start is not None:
        spans.add((active_label, active_start, len(label_ids)))
    return spans


class _OnnxRuntimeWrapper:
    """Duck-types just enough of HF's model interface for benchmark code:
    .config.id2label and __call__(input_ids=..., attention_mask=...) -> obj with .logits."""

    def __init__(self, model_path: str, *, use_cuda: bool):
        import os
        import onnxruntime as ort

        # Discover .onnx file (model.onnx by convention)
        candidates = [
            os.path.join(model_path, "model.onnx"),
            os.path.join(model_path, "model_quantized.onnx"),
        ]
        onnx_file = next((c for c in candidates if os.path.isfile(c)), None)
        if onnx_file is None:
            raise FileNotFoundError(f"No model.onnx found under {model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(onnx_file, sess_options=sess_opts, providers=providers)
        # Load HF config from the same dir for id2label.
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self._input_names = {i.name for i in self.session.get_inputs()}

    def __call__(self, **kwargs):
        # Accept torch tensors; convert to numpy on CPU.
        feed = {}
        for name, val in kwargs.items():
            if name not in self._input_names:
                continue
            if hasattr(val, "detach"):
                val = val.detach().cpu().numpy()
            feed[name] = val
        outputs = self.session.run(None, feed)
        # Wrap in object exposing .logits as a torch tensor (for downstream .float()/.argmax compat).
        logits = torch.as_tensor(outputs[0])
        return type("OnnxOut", (), {"logits": logits})()


def load_model(model_path: str, *, onnx: bool = False, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    use_cuda = (device == "cuda") or (device == "auto" and torch.cuda.is_available())

    if onnx:
        model = _OnnxRuntimeWrapper(model_path, use_cuda=use_cuda)
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        model.eval()
        if use_cuda:
            model = model.to("cuda")

    id_to_label = _normalize_id2label(model.config.id2label)
    categories = _extract_categories_from_id2label(id_to_label)
    return tokenizer, model, id_to_label, categories, {"onnx": onnx, "cuda": use_cuda}


def predict_spans(
    tokenizer,
    model,
    id_to_label: dict[int, str],
    text: str,
    *,
    max_length: int,
    allowed_labels: set[str],
    model_info: dict | None = None,
) -> set[tuple[str, int, int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    info = model_info or {}
    is_onnx = bool(info.get("onnx"))
    on_cuda = bool(info.get("cuda"))
    # ONNX inputs stay on CPU (the CUDAExecutionProvider handles transfer internally).
    if not is_onnx and on_cuda:
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
    predictions = (
        outputs.logits.float().detach().cpu().numpy().argmax(axis=-1)[0].tolist()
    )
    token_spans = _decode_bioes_spans(predictions, id_to_label)

    pred_set: set[tuple[str, int, int]] = set()
    for label, token_start, token_end in token_spans:
        if label not in allowed_labels:
            continue
        if token_end <= token_start:
            continue
        start_char = int(offset_mapping[token_start][0])
        end_char = int(offset_mapping[token_end - 1][1])
        if end_char <= start_char:
            continue
        pred_set.add((label, start_char, end_char))
    return pred_set


def _calc_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def benchmark_model(
    *,
    model_name: str,
    model_path: str,
    records: list[dict[str, Any]],
    benchmark_labels: list[str],
    max_length: int,
    onnx: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    print(f"Loading {model_name}: {model_path}", flush=True)
    tokenizer, model, id_to_label, supported_labels, model_info = load_model(
        model_path, onnx=onnx, device=device
    )
    print(
        f"  supported labels ({len(supported_labels)}): {sorted(supported_labels)}  "
        f"backend={'onnx' if model_info['onnx'] else 'pytorch'}  "
        f"device={'cuda' if model_info['cuda'] else 'cpu'}",
        flush=True,
    )

    allowed_labels = set(benchmark_labels)
    overall_tp = overall_fp = overall_fn = 0
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()
    latencies_ms: list[float] = []

    t0 = time.time()
    for i, record in enumerate(records):
        text = str(record.get("text", ""))
        gt_set = _normalize_span_set(
            text,
            _extract_gt_spans(record, allowed_labels=allowed_labels),
        )
        t_pred_start = time.perf_counter()
        pred_set = predict_spans(
            tokenizer,
            model,
            id_to_label,
            text,
            max_length=max_length,
            allowed_labels=allowed_labels,
            model_info=model_info,
        )
        latencies_ms.append((time.perf_counter() - t_pred_start) * 1000.0)
        pred_set = _normalize_span_set(text, pred_set)

        matched = gt_set & pred_set
        overall_tp += len(matched)
        overall_fp += len(pred_set - gt_set)
        overall_fn += len(gt_set - pred_set)

        for label, start, end in gt_set:
            if (label, start, end) in pred_set:
                per_label_tp[label] += 1
            else:
                per_label_fn[label] += 1
        for label, start, end in pred_set:
            if (label, start, end) not in gt_set:
                per_label_fp[label] += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{model_name}] {i + 1}/{len(records)} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    total_gold = overall_tp + overall_fn
    total_pred = overall_tp + overall_fp

    overall = {
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "gold_spans": total_gold,
        "pred_spans": total_pred,
        **_calc_prf(overall_tp, overall_fp, overall_fn),
    }

    per_label = {}
    for label in benchmark_labels:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_spans": tp + fn,
            "pred_spans": tp + fp,
            **_calc_prf(tp, fp, fn),
        }

    speed = {}
    if latencies_ms:
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)
        # skip first 5 as warmup if we have enough samples
        warmup = min(5, max(0, n - 20))
        steady = sorted_lat[warmup:] if warmup else sorted_lat
        steady_unsorted = latencies_ms[warmup:] if warmup else latencies_ms
        speed = {
            "device": "cuda" if model_info["cuda"] else "cpu",
            "backend": "onnx" if model_info["onnx"] else "pytorch",
            "n_warmup": warmup,
            "n_measured": len(steady),
            "latency_ms_mean": round(sum(steady_unsorted) / len(steady_unsorted), 2),
            "latency_ms_p50": round(steady[len(steady) // 2], 2),
            "latency_ms_p95": round(steady[max(0, int(0.95 * len(steady)) - 1)], 2),
            "latency_ms_p99": round(steady[max(0, int(0.99 * len(steady)) - 1)], 2),
            "throughput_rps": round(1000.0 / (sum(steady_unsorted) / len(steady_unsorted)), 2),
        }

    return {
        "model_name": model_name,
        "model_path": model_path,
        "supported_labels": sorted(supported_labels),
        "benchmark_labels": benchmark_labels,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / len(records), 1) if records else 0.0,
        "overall": overall,
        "per_label": per_label,
        "speed": speed,
    }


def _build_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 2:
        return {}

    baseline = results[0]
    baseline_name = baseline["model_name"]
    comparison: dict[str, Any] = {
        "baseline_model": baseline_name,
        "vs_baseline": {},
    }

    for candidate in results[1:]:
        candidate_name = candidate["model_name"]
        comparison["vs_baseline"][candidate_name] = {
            "overall_f1_delta": round(
                candidate["overall"]["f1"] - baseline["overall"]["f1"], 4
            ),
            "overall_precision_delta": round(
                candidate["overall"]["precision"] - baseline["overall"]["precision"], 4
            ),
            "overall_recall_delta": round(
                candidate["overall"]["recall"] - baseline["overall"]["recall"], 4
            ),
            "per_label_f1_delta": {
                label: round(
                    candidate["per_label"][label]["f1"] - baseline["per_label"][label]["f1"],
                    4,
                )
                for label in baseline["benchmark_labels"]
            },
        }
    return comparison


def _print_summary(results: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    print(f"\n{'=' * 96}")
    print(
        f"{'MODEL':<24} {'SPAN F1':>8} {'PREC':>8} {'REC':>8} "
        f"{'GOLD':>8} {'PRED':>8} {'TP':>8} {'FP':>8} {'FN':>8}"
    )
    print("-" * 96)
    for result in results:
        overall = result["overall"]
        print(
            f"{result['model_name']:<24} {overall['f1']:>8.4f} {overall['precision']:>8.4f} "
            f"{overall['recall']:>8.4f} {overall['gold_spans']:>8} {overall['pred_spans']:>8} "
            f"{overall['tp']:>8} {overall['fp']:>8} {overall['fn']:>8}"
        )
    print("=" * 96)

    for result in results:
        print(f"\n  [{result['model_name']}] Per-label span metrics:")
        print(
            f"  {'LABEL':<20} {'F1':>8} {'PREC':>8} {'REC':>8} "
            f"{'GOLD':>8} {'PRED':>8}"
        )
        print(f"  {'-' * 64}")
        for label, stats in sorted(
            result["per_label"].items(),
            key=lambda item: -(item[1]["gold_spans"]),
        ):
            print(
                f"  {label:<20} {stats['f1']:>8.4f} {stats['precision']:>8.4f} "
                f"{stats['recall']:>8.4f} {stats['gold_spans']:>8} {stats['pred_spans']:>8}"
            )

    if comparison:
        baseline_name = comparison["baseline_model"]
        print(f"\n  Baseline: {baseline_name}")
        for candidate_name, stats in comparison["vs_baseline"].items():
            print(
                f"  {candidate_name}: "
                f"delta_f1={stats['overall_f1_delta']:+.4f} "
                f"delta_precision={stats['overall_precision_delta']:+.4f} "
                f"delta_recall={stats['overall_recall_delta']:+.4f}"
            )


def main() -> None:
    args = parse_args()

    model_paths = args.model_paths
    model_names = _resolve_model_names(model_paths, args.resolved_model_names)
    dataset_path = Path(args.dataset)
    label_space_path = Path(args.label_space_json)

    dataset_labels = _load_dataset_labels(label_space_path)
    records = _load_jsonl(dataset_path)
    records = _sample_records(records, max_samples=args.samples, seed=args.seed)

    per_model_supported = {
        model_name: _load_model_categories(model_path)
        for model_name, model_path in zip(model_names, model_paths, strict=True)
    }
    if args.labels_mode == "shared":
        benchmark_labels = sorted(
            set(dataset_labels).intersection(*per_model_supported.values())
        )
    else:
        benchmark_labels = sorted(dataset_labels)

    if not benchmark_labels:
        raise ValueError("No benchmark labels remain after applying labels-mode")

    excluded_dataset_labels = sorted(set(dataset_labels) - set(benchmark_labels))

    print(f"Dataset: {dataset_path}")
    print(f"Label space: {label_space_path}")
    print(f"Records: {len(records)}")
    print(f"Labels mode: {args.labels_mode}")
    print(f"Benchmark labels ({len(benchmark_labels)}): {benchmark_labels}")
    if excluded_dataset_labels:
        print(f"Excluded dataset labels: {excluded_dataset_labels}")
    print()

    results = []
    for model_name, model_path in zip(model_names, model_paths, strict=True):
        results.append(
            benchmark_model(
                model_name=model_name,
                model_path=model_path,
                records=records,
                benchmark_labels=benchmark_labels,
                max_length=args.max_length,
                onnx=args.onnx,
                device=args.device,
            )
        )

    comparison = _build_comparison(results)
    _print_summary(results, comparison)

    payload = {
        "dataset": str(dataset_path),
        "label_space_json": str(label_space_path),
        "labels_mode": args.labels_mode,
        "benchmark_labels": benchmark_labels,
        "excluded_dataset_labels": excluded_dataset_labels,
        "records": len(records),
        "results": results,
        "comparison": comparison,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
