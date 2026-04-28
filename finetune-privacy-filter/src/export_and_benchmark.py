#!/usr/bin/env python3
"""
Export the fine-tuned privacy filter to multiple ONNX precisions, benchmark each
on both CPU and GPU, and drop variants whose F1 deviates beyond a threshold.

------------------------------------------------------------------------------
STATUS: BLOCKED (as of 2026-04-28). Kept here as a record of the scaffolding
required once `openai_privacy_filter` becomes ONNX-exportable.

The custom `OpenAIPrivacyFilterForTokenClassification` modeling code (sparse
MoE, gpt-oss-style) is not currently exportable via the public toolchain:

  * Legacy tracer path (`torch.onnx.export` / optimum non-dynamo) hits
    `sdpa_mask_without_vmap() missing 1 required positional argument:
    'cache_position'` — optimum-onnx PR #113's transformers 5.x compat shim
    does not cover this code path.
  * Dynamo path (`torch.export.export` / optimum `dynamo=True`, requires
    torch>=2.9) hits `TypeError: missing a required argument: 'self'` on the
    decorator-wrapped MoE forward — a model-source issue, not tooling.

Patching `modeling_openai_privacy_filter.py` to be `torch.export`-compatible
is the unblocker; the base model's published ONNX exports were almost
certainly produced with internal tooling.

The Dockerfile + compose service + this script + `_OnnxRuntimeWrapper` in
`benchmark_pii_heldout.py` are wired end-to-end and only the `main_export`
call fails. Resume here once the modeling code is exportable.
------------------------------------------------------------------------------

Variants attempted: onnx_fp16, onnx_int8.
Baseline: the input PyTorch (bf16) model on GPU.

Outputs into PF_EXPORT_OUTPUT/:
  baseline_benchmark.json          # bf16 PyTorch on GPU (reference F1 + speed)
  baseline_benchmark_cpu.json      # bf16 PyTorch on CPU
  onnx_fp16/                       # kept variant: model + benchmark_{gpu,cpu}.json
  onnx_int8/                       # kept variant
  export_summary.json              # consolidated table of F1 deltas + speed

Usage (env-driven; defaults match the train-pii output layout):
  PF_EXPORT_INPUT       trained model dir              (default: data/checkpoints/ko_pii_hf_ddp_v6_lora)
  PF_EXPORT_OUTPUT      output dir                     (default: data/exports)
  PF_EXPORT_DATASET     test JSONL                     (default: data/generated/ko_pii_opf_v4/test.jsonl)
  PF_EXPORT_LABEL_SPACE label_space.json               (default: data/generated/ko_pii_opf_v4/label_space.json)
  PF_EXPORT_MAX_F1_DROP per-label drop threshold       (default: 0.02)
  PF_EXPORT_MAX_LENGTH  tokenizer max length           (default: 512)
  PF_EXPORT_VARIANTS    comma-separated subset         (default: fp16,int8)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


VARIANT_NAMES = ("fp16", "int8")


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _copy_metadata(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer + config + label space from input dir to output dir."""
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "config.json", "label_space.json"):
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def _build_custom_onnx_config(input_model: Path):
    """openai_privacy_filter is a custom architecture; optimum doesn't ship a config for it.
    Construct a minimal OnnxConfig for token classification: input_ids + attention_mask -> logits."""
    from optimum.exporters.onnx.base import OnnxConfig
    from optimum.utils import NormalizedTextConfig, DummyTextInputGenerator
    from transformers import AutoConfig

    class OpenAIPrivacyFilterOnnxConfig(OnnxConfig):
        DEFAULT_ONNX_OPSET = 17
        NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
        DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

        @property
        def inputs(self) -> dict[str, dict[int, str]]:
            return {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }

        @property
        def outputs(self) -> dict[str, dict[int, str]]:
            return {"logits": {0: "batch_size", 1: "sequence_length"}}

    cfg = AutoConfig.from_pretrained(str(input_model), trust_remote_code=True)
    return OpenAIPrivacyFilterOnnxConfig(cfg, task="token-classification")


def _export_fp32_onnx(input_model: Path, out_dir: Path, max_length: int) -> Path:
    """Export via optimum + dynamo path. Dynamo is required because the legacy torch.onnx
    tracer hits a transformers 5.x masking_utils incompatibility. Needs torch>=2.9."""
    from optimum.exporters.onnx import main_export

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_config = _build_custom_onnx_config(input_model)
    print(f"  optimum main_export(dynamo=True) -> {out_dir}", flush=True)
    main_export(
        model_name_or_path=str(input_model),
        output=str(out_dir),
        task="token-classification",
        trust_remote_code=True,
        device="cpu",
        custom_onnx_configs={"model": onnx_config},
        dynamo=True,
        do_validation=False,
    )
    _copy_metadata(input_model, out_dir)
    onnx_files = list(out_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"export produced no .onnx file in {out_dir}")
    return onnx_files[0]


def export_fp16(input_model: Path, out_dir: Path, max_length: int) -> None:
    """Export fp32 then convert to fp16 with onnxconverter-common."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    fp32_tmp = out_dir.parent / f"_fp32_{out_dir.name}_tmp"
    if fp32_tmp.exists():
        shutil.rmtree(fp32_tmp)

    fp32_path = _export_fp32_onnx(input_model, fp32_tmp, max_length)

    print("  converting fp32 -> fp16 ...", flush=True)
    import onnx
    from onnxconverter_common import float16

    out_dir.mkdir(parents=True, exist_ok=True)
    model_fp32 = onnx.load(str(fp32_path), load_external_data=True)
    model_fp16 = float16.convert_float_to_float16(
        model_fp32, keep_io_types=True, disable_shape_infer=True
    )
    onnx.save(model_fp16, str(out_dir / "model.onnx"),
              save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx_data", convert_attribute=False)
    _copy_metadata(input_model, out_dir)
    shutil.rmtree(fp32_tmp, ignore_errors=True)


def export_int8(input_model: Path, out_dir: Path, max_length: int, scratch: Path) -> None:
    """Export fp32 then dynamic-quantize int8 weights via onnxruntime.quantization."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    fp32_tmp = scratch / "_fp32_int8_source"
    if fp32_tmp.exists():
        shutil.rmtree(fp32_tmp)

    fp32_path = _export_fp32_onnx(input_model, fp32_tmp, max_length)

    print("  dynamic int8 quantization ...", flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    out_dir.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_dir / "model.onnx"),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    _copy_metadata(input_model, out_dir)
    shutil.rmtree(fp32_tmp, ignore_errors=True)


def run_benchmark(
    *,
    model_path: Path,
    dataset: Path,
    label_space: Path,
    output_json: Path,
    max_length: int,
    onnx: bool,
    device: str,
    model_name: str,
) -> dict[str, Any]:
    """Invoke benchmark_pii_heldout.py as a subprocess and return parsed JSON."""
    cmd = [
        sys.executable, "/workspace/benchmark_pii_heldout.py",
        "--models", str(model_path),
        "--model-names", model_name,
        "--dataset", str(dataset),
        "--label-space-json", str(label_space),
        "--max-length", str(max_length),
        "--device", device,
        "--output", str(output_json),
    ]
    if onnx:
        cmd.append("--onnx")
    run(cmd)
    with output_json.open() as f:
        return json.load(f)


def per_label_f1(result_payload: dict) -> dict[str, float]:
    """Pull {label: f1} from a benchmark JSON (single model)."""
    models = result_payload.get("results") or []
    if not models:
        return {}
    pl = models[0].get("per_label", {})
    return {label: float(metrics.get("f1", 0.0)) for label, metrics in pl.items()}


def overall_f1(result_payload: dict) -> float:
    models = result_payload.get("results") or []
    if not models:
        return 0.0
    return float(models[0].get("overall", {}).get("f1", 0.0))


def speed_summary(result_payload: dict) -> dict:
    models = result_payload.get("results") or []
    if not models:
        return {}
    return models[0].get("speed", {}) or {}


def compare_f1(baseline: dict[str, float], variant: dict[str, float]) -> tuple[float, str]:
    """Returns (max_drop, label_with_max_drop). Drop is positive when variant < baseline."""
    max_drop = 0.0
    worst_label = ""
    all_labels = set(baseline) | set(variant)
    for lbl in all_labels:
        b = baseline.get(lbl, 0.0)
        v = variant.get(lbl, 0.0)
        drop = b - v
        if drop > max_drop:
            max_drop = drop
            worst_label = lbl
    return round(max_drop, 4), worst_label


def dir_size_mb(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total // (1024 * 1024)


def main() -> int:
    input_model = env_path("PF_EXPORT_INPUT",
                           "/workspace/data/checkpoints/ko_pii_hf_ddp_v6_lora")
    output_dir = env_path("PF_EXPORT_OUTPUT", "/workspace/data/exports")
    dataset = env_path("PF_EXPORT_DATASET",
                       "/workspace/data/generated/ko_pii_opf_v4/test.jsonl")
    label_space = env_path("PF_EXPORT_LABEL_SPACE",
                           "/workspace/data/generated/ko_pii_opf_v4/label_space.json")
    max_f1_drop = env_float("PF_EXPORT_MAX_F1_DROP", 0.02)
    max_length = env_int("PF_EXPORT_MAX_LENGTH", 512)
    variants = env_list("PF_EXPORT_VARIANTS", list(VARIANT_NAMES))

    print(f"input_model:  {input_model}")
    print(f"output_dir:   {output_dir}")
    print(f"dataset:      {dataset}")
    print(f"label_space:  {label_space}")
    print(f"variants:     {variants}")
    print(f"max_f1_drop:  {max_f1_drop}")
    print(f"max_length:   {max_length}")

    if not input_model.exists():
        print(f"ERROR: input model not found: {input_model}", file=sys.stderr)
        return 2
    if not dataset.exists() or not label_space.exists():
        print(f"ERROR: dataset/label_space not found", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # === Baseline: bf16 PyTorch, GPU then CPU ===
    print("\n=== BASELINE: bf16 PyTorch ===")
    baseline_gpu_path = output_dir / "baseline_benchmark.json"
    baseline_gpu = run_benchmark(
        model_path=input_model, dataset=dataset, label_space=label_space,
        output_json=baseline_gpu_path, max_length=max_length,
        onnx=False, device="cuda", model_name="baseline_bf16_gpu",
    )
    baseline_cpu_path = output_dir / "baseline_benchmark_cpu.json"
    baseline_cpu = run_benchmark(
        model_path=input_model, dataset=dataset, label_space=label_space,
        output_json=baseline_cpu_path, max_length=max_length,
        onnx=False, device="cpu", model_name="baseline_bf16_cpu",
    )
    baseline_f1 = per_label_f1(baseline_gpu)
    baseline_overall = overall_f1(baseline_gpu)
    print(f"\nBaseline overall F1 (GPU bf16): {baseline_overall:.4f}")

    # === Variants ===
    summaries = []
    for variant in variants:
        if variant not in VARIANT_NAMES:
            print(f"WARN: unknown variant '{variant}', skipping")
            continue

        variant_dir = output_dir / f"onnx_{variant}"
        print(f"\n=== EXPORTING onnx_{variant} ===")
        try:
            if variant == "fp16":
                export_fp16(input_model, variant_dir, max_length)
            elif variant == "int8":
                export_int8(input_model, variant_dir, max_length, scratch=output_dir)
        except Exception as exc:
            print(f"ERROR exporting {variant}: {exc}")
            summaries.append({
                "name": f"onnx_{variant}", "status": "EXPORT_FAILED",
                "kept": False, "reason": str(exc),
            })
            continue

        size_mb = dir_size_mb(variant_dir)

        # Benchmark on GPU
        gpu_path = variant_dir / "benchmark_gpu.json"
        try:
            gpu_result = run_benchmark(
                model_path=variant_dir, dataset=dataset, label_space=label_space,
                output_json=gpu_path, max_length=max_length,
                onnx=True, device="cuda", model_name=f"onnx_{variant}_gpu",
            )
            gpu_f1 = per_label_f1(gpu_result)
            gpu_overall_f1 = overall_f1(gpu_result)
            gpu_speed = speed_summary(gpu_result)
        except Exception as exc:
            print(f"ERROR benchmarking {variant} on GPU: {exc}")
            gpu_f1, gpu_overall_f1, gpu_speed = {}, 0.0, {}

        # Benchmark on CPU
        cpu_path = variant_dir / "benchmark_cpu.json"
        try:
            cpu_result = run_benchmark(
                model_path=variant_dir, dataset=dataset, label_space=label_space,
                output_json=cpu_path, max_length=max_length,
                onnx=True, device="cpu", model_name=f"onnx_{variant}_cpu",
            )
            cpu_speed = speed_summary(cpu_result)
        except Exception as exc:
            print(f"ERROR benchmarking {variant} on CPU: {exc}")
            cpu_speed = {}

        # F1 comparison (GPU run, since F1 should be device-invariant; GPU is faster).
        max_drop, worst_label = compare_f1(baseline_f1, gpu_f1)
        overall_drop = round(baseline_overall - gpu_overall_f1, 4)
        kept = max_drop <= max_f1_drop and overall_drop <= max_f1_drop

        summary_entry = {
            "name": f"onnx_{variant}",
            "status": "PASS" if kept else "FAIL",
            "kept": kept,
            "size_mb": size_mb,
            "overall_f1_baseline": round(baseline_overall, 4),
            "overall_f1_variant": round(gpu_overall_f1, 4),
            "overall_f1_drop": overall_drop,
            "max_label_drop": max_drop,
            "worst_label": worst_label,
            "f1_per_label": {
                label: {
                    "baseline": round(baseline_f1.get(label, 0.0), 4),
                    "variant":  round(gpu_f1.get(label, 0.0), 4),
                    "drop":     round(baseline_f1.get(label, 0.0) - gpu_f1.get(label, 0.0), 4),
                }
                for label in sorted(set(baseline_f1) | set(gpu_f1))
            },
            "speed_gpu": gpu_speed,
            "speed_cpu": cpu_speed,
        }
        if not kept:
            summary_entry["reason"] = (
                f"{worst_label} dropped {max_drop:.4f} (max allowed {max_f1_drop:.4f})"
                if max_drop > max_f1_drop
                else f"overall dropped {overall_drop:.4f} (max allowed {max_f1_drop:.4f})"
            )
            print(f"\n❌ DROPPING onnx_{variant}: {summary_entry['reason']}")
            shutil.rmtree(variant_dir, ignore_errors=True)
        else:
            print(f"\n✅ KEEPING onnx_{variant}: max label drop {max_drop:.4f}, size {size_mb}MB")

        summaries.append(summary_entry)

    # === Final summary ===
    summary = {
        "input_model": str(input_model),
        "dataset": str(dataset),
        "max_f1_drop_threshold": max_f1_drop,
        "baseline": {
            "format": "bf16-pytorch",
            "overall_f1": round(baseline_overall, 4),
            "f1_per_label": {k: round(v, 4) for k, v in sorted(baseline_f1.items())},
            "speed_gpu": speed_summary(baseline_gpu),
            "speed_cpu": speed_summary(baseline_cpu),
        },
        "variants": summaries,
    }
    summary_path = output_dir / "export_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"Baseline (bf16 GPU):  F1 {baseline_overall:.4f}")
    if summary["baseline"]["speed_gpu"]:
        print(f"  GPU latency p50: {summary['baseline']['speed_gpu'].get('latency_ms_p50','?')}ms")
    if summary["baseline"]["speed_cpu"]:
        print(f"  CPU latency p50: {summary['baseline']['speed_cpu'].get('latency_ms_p50','?')}ms")
    for v in summaries:
        status = "PASS" if v.get("kept") else "FAIL"
        print(f"\n  {v['name']:20s} [{status}]  size {v.get('size_mb','?')}MB")
        print(f"    overall F1: {v.get('overall_f1_variant','?')}  (drop {v.get('overall_f1_drop','?')})")
        print(f"    max-label drop: {v.get('max_label_drop','?')} ({v.get('worst_label','?')})")
        sg = v.get("speed_gpu", {})
        sc = v.get("speed_cpu", {})
        if sg:
            print(f"    GPU p50 {sg.get('latency_ms_p50','?')}ms  ({sg.get('throughput_rps','?')} rps)")
        if sc:
            print(f"    CPU p50 {sc.get('latency_ms_p50','?')}ms  ({sc.get('throughput_rps','?')} rps)")
        if not v.get("kept"):
            print(f"    REASON: {v.get('reason','')}")

    print(f"\nFull summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
