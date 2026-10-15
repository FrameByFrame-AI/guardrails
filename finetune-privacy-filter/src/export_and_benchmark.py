#!/usr/bin/env python3
"""
Export the fine-tuned privacy filter to multiple ONNX precisions, benchmark each
on both CPU and GPU, and drop variants whose F1 deviates beyond a threshold.

------------------------------------------------------------------------------
STATUS: UNBLOCKED (2026-07-04). Both previously-blocking export errors are
resolved:

  * Legacy tracer path (`torch.onnx.export` / optimum non-dynamo) used to hit
    `sdpa_mask_without_vmap() missing 1 required positional argument:
    'cache_position'`. This is gone as of a fresh `transformers` install
    (upstream removed `cache_position` from masking primitives specifically
    because it broke ONNX export). optimum's exporter no longer even
    attempts this path — it goes straight to dynamo.
  * Dynamo path (`torch.export.export` / optimum `dynamo=True`) hit three
    *sequential* errors once each prior one was fixed — all resolved now:

    1. `TypeError: missing a required argument: 'self'` on decorated
       `forward` methods (`OpenAIPrivacyFilterExperts` and others). Root
       cause: several decorators in `transformers/integrations/moe.py` and
       `transformers/utils/generic.py` (`@use_experts_implementation`,
       `@merge_with_config_defaults`, `@capture_outputs`, ...) rewrap
       `forward` as `forward(self, *args, **kwargs)` but apply
       `@functools.wraps(original_forward)`, which copies `original_forward`'s
       *named* signature onto the wrapper via `__wrapped__` without the
       wrapper actually accepting those names positionally in its own
       bytecode. `torch.export`'s `_combine_args` calls
       `inspect.signature(nn_module.forward).bind(...)`, which follows
       `__wrapped__` and gets a signature that doesn't match what the
       wrapper function actually accepts, dropping `self`. Simply stripping
       `__wrapped__` isn't enough either — `optimum`'s own
       `check_dummy_inputs_are_allowed` preflight check *needs* the named
       signature to match input names against the ONNX config. The real fix,
       in `_rebuild_forward_with_real_signature`, synthesizes a real
       bytecode-level signature (via `exec`) that forwards positionally to
       the original dispatch logic — satisfying both consumers.
    2. `TypeError: sdpa_mask_without_vmap() missing 1 required positional
       argument: 'cache_position'`. `optimum-onnx`'s vendored, vmap-free
       reimplementation of this masking function still requires
       `cache_position`; current `transformers` no longer passes one (removed
       upstream *specifically because it broke ONNX export*, and replaced
       with `q_length`/`q_offset` instead). `_patch_optimum_mask_cache_position`
       synthesizes `cache_position` from those on the way in.
    3. `GuardOnDataDependentSymNode` on `for expert_idx in expert_hit:` inside
       the default eager MoE experts forward — a Python loop whose length
       depends on which experts a given input actually routes to, which is
       inherently data-dependent and untraceable. `transformers.integrations.moe`
       registers alternative vectorized backends (`batched_mm`, `grouped_mm`,
       ...) specifically to avoid this control flow;
       `_prepare_export_source` sets `config._experts_implementation =
       "batched_mm"` (dense, masked, no data-dependent loop or sort) on a
       scratch copy of the checkpoint before export.

    All three patches are process-local (applied to the already-imported
    module/classes at runtime) and are not permanent edits to the installed
    `transformers`/`optimum` packages. (1) and (2) look like real upstream
    bugs worth reporting; (3) is closer to "this model's default eager path
    was never meant to be exported," and `batched_mm` is a legitimate
    fallback already provided by transformers for MoE export/compile.
------------------------------------------------------------------------------

Variants attempted: onnx_fp32 (works), onnx_fp16, onnx_int8 (both still blocked
-- see STATUS notes on export_fp16/export_int8 above).
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
  PF_EXPORT_EXPERTS_IMPLEMENTATION  MoE backend for export (default: batched_mm)
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


VARIANT_NAMES = ("fp32", "fp16", "int8")


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


def _rebuild_forward_with_real_signature(cls: type) -> None:
    """Replace `cls.forward` with a passthrough wrapper whose *actual*
    bytecode signature matches its `__wrapped__`-reported one, instead of
    being a bare `(self, *args, **kwargs)` dispatcher wearing a borrowed
    signature. See `_patch_moe_experts_for_export` for why this is needed.
    """
    original_forward = cls.__dict__.get("forward")
    if original_forward is None or not hasattr(original_forward, "__wrapped__"):
        return
    code = getattr(original_forward, "__code__", None)
    if code is None or code.co_argcount > 1:
        return  # already has a real multi-arg signature; nothing to fix

    sig = inspect.signature(original_forward)
    params = list(sig.parameters.values())
    if not params or params[0].name != "self":
        return

    # Call positionally, not by keyword: `config._experts_implementation` can
    # swap in a different backend function (e.g. `batched_mm_experts_forward`)
    # that uses different parameter *names* for the same positional slots
    # (`top_k_index`/`top_k_weights` vs. this class's own `router_indices`/
    # `routing_weights`). Positional forwarding stays correct across backends;
    # keyword forwarding would silently break whichever backend doesn't use
    # this class's exact names.
    namespace: dict[str, Any] = {"__original_forward": original_forward}
    arg_defs = []
    call_parts = []
    for i, p in enumerate(params[1:]):
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            arg_defs.append(f"**{p.name}")
            call_parts.append(f"**{p.name}")
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            arg_defs.append(f"*{p.name}")
            call_parts.append(f"*{p.name}")
        elif p.default is inspect.Parameter.empty:
            arg_defs.append(p.name)
            call_parts.append(p.name)
        else:
            default_name = f"__default_{i}"
            namespace[default_name] = p.default
            arg_defs.append(f"{p.name}={default_name}")
            call_parts.append(p.name)

    src = (
        f"def _forward(self, {', '.join(arg_defs)}):\n"
        f"    return __original_forward(self, {', '.join(call_parts)})\n"
    )
    exec(src, namespace)  # noqa: S102 -- building a signature-preserving passthrough, not eval'ing input
    cls.forward = namespace["_forward"]


def _patch_moe_experts_for_export() -> None:
    """Fix every decorated `forward` in the openai_privacy_filter modeling
    module whose real signature is `(self, *args, **kwargs)` but whose
    `__wrapped__` (from `@functools.wraps`) reports a different, named one.

    Several decorators in this module (`@use_experts_implementation`,
    `@merge_with_config_defaults`, `@capture_outputs`, `@dynamic_rope_update`,
    ...) rewrap `forward` as `forward(self, *args, **kwargs)` but apply
    `@functools.wraps(original_forward)`, which copies `original_forward`'s
    *named* signature onto the wrapper via `__wrapped__` without the wrapper
    actually accepting those names positionally. Two different consumers
    disagree about which signature they want:

      - `torch.export`'s `_combine_args` calls
        `inspect.signature(nn_module.forward).bind(...)`, which follows
        `__wrapped__` and gets a signature that doesn't match what the
        wrapper function actually accepts — dropping `self` from the bound
        arguments and raising `TypeError: missing a required argument:
        'self'`.
      - `optimum`'s `check_dummy_inputs_are_allowed` also inspects the
        signature, but needs the *named* one (`input_ids`, `attention_mask`,
        ...) to match against the declared ONNX config inputs — simply
        stripping `__wrapped__` satisfies torch.export but then fails this
        check instead, since the bare wrapper's real signature is just
        `(self, *args, **kwargs)`.

    The actual fix is to give these `forward` methods a real, named
    bytecode-level signature (via `_rebuild_forward_with_real_signature`)
    that still dispatches to the original wrapped logic, satisfying both
    consumers at once. Detected via `co_argcount <= 1` (only `self` is a real
    positional parameter in the compiled bytecode) rather than hardcoding a
    class list that will drift as the model changes.

    Process-local; does not touch the installed package on disk.
    """
    import transformers.models.openai_privacy_filter.modeling_openai_privacy_filter as opf_modeling

    for _, obj in vars(opf_modeling).items():
        if isinstance(obj, type):
            _rebuild_forward_with_real_signature(obj)


def _patch_optimum_mask_cache_position() -> None:
    """optimum-onnx's vendored `sdpa_mask_without_vmap` (a from-scratch
    reimplementation of transformers' masking function, kept vmap-free for
    ONNX export) still requires a `cache_position` argument. Current
    `transformers` (`create_bidirectional_sliding_window_mask`) no longer
    passes one — it was removed from the masking call signature upstream,
    specifically because it broke ONNX export — and calls the mask interface
    with `q_length`/`q_offset` instead. optimum-onnx's shim hasn't caught up,
    so the call arrives without `cache_position` and raises `TypeError:
    sdpa_mask_without_vmap() missing 1 required positional argument:
    'cache_position'`.

    `cache_position` and `q_length`/`q_offset` describe the same thing (query
    token position indices) in the two APIs, so this wraps
    `sdpa_mask_without_vmap` to synthesize `cache_position =
    arange(q_offset, q_offset + q_length)` when it's missing but the modern
    keys are present. Process-local; does not touch the installed package.
    """
    import torch
    from optimum.exporters.onnx import model_patcher

    original = model_patcher.sdpa_mask_without_vmap

    def patched(*args, **kwargs):
        if "cache_position" not in kwargs and len(args) < 2:
            q_length = kwargs.pop("q_length", None)
            q_offset = kwargs.pop("q_offset", 0)
            if q_length is not None:
                attention_mask = kwargs.get("attention_mask")
                device = attention_mask.device if attention_mask is not None else None
                kwargs["cache_position"] = torch.arange(
                    q_offset, q_offset + q_length, device=device
                )
        return original(*args, **kwargs)

    model_patcher.sdpa_mask_without_vmap = patched


def _build_custom_onnx_config(input_model: Path):
    """openai_privacy_filter is a custom architecture; optimum doesn't ship a config for it.
    Construct a minimal OnnxConfig for token classification: input_ids + attention_mask -> logits."""
    from optimum.exporters.onnx.base import OnnxConfig
    from optimum.utils import NormalizedTextConfig, DummyTextInputGenerator
    from transformers import AutoConfig

    class OpenAIPrivacyFilterOnnxConfig(OnnxConfig):
        # opset 17 conversion fails in onnxscript's version converter for this
        # graph ("No initializer or constant input to node found" in the axes
        # adapter); the exporter falls back to keeping the native opset (18)
        # with a warning either way, so target it directly and skip the noise.
        DEFAULT_ONNX_OPSET = 18
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


def _prepare_export_source(input_model: Path, scratch_dir: Path, experts_implementation: str) -> Path:
    """Symlink-copy `input_model` into `scratch_dir` with `config._experts_implementation`
    set, so `main_export` loads a model that dispatches MoE experts through a
    vectorized backend instead of the default eager one.

    The default eager `OpenAIPrivacyFilterExperts` forward does
    `for expert_idx in expert_hit: ...` — a Python loop whose length depends
    on which experts a given input actually routes to. That's genuinely
    data-dependent control flow: `torch.export` can't trace "iterate N times"
    when N is only known from tensor *values* at run time, and fails with
    `GuardOnDataDependentSymNode`. `transformers.integrations.moe` registers
    alternative vectorized implementations (`batched_mm`, `grouped_mm`, ...)
    specifically to avoid this; `batched_mm` computes all experts densely
    with masking (no data-dependent loop, no sorting-by-expert-count either),
    which is the most export-safe of the available options. Uses symlinks to
    avoid duplicating the multi-GB safetensors weights.
    """
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    for item in input_model.iterdir():
        if item.name == "config.json":
            continue
        (scratch_dir / item.name).symlink_to(os.path.relpath(item, scratch_dir))

    config_payload = json.loads((input_model / "config.json").read_text(encoding="utf-8"))
    config_payload["_experts_implementation"] = experts_implementation
    (scratch_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return scratch_dir


def _export_fp32_onnx(input_model: Path, out_dir: Path, max_length: int) -> Path:
    """Export via optimum + dynamo path. Dynamo is required because the legacy torch.onnx
    tracer hits a transformers 5.x masking_utils incompatibility. Needs torch>=2.9."""
    from optimum.exporters.onnx import main_export

    _patch_moe_experts_for_export()
    _patch_optimum_mask_cache_position()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_source = _prepare_export_source(
        input_model,
        out_dir.parent / f"_export_source_{out_dir.name}",
        experts_implementation=os.environ.get("PF_EXPORT_EXPERTS_IMPLEMENTATION", "batched_mm"),
    )
    onnx_config = _build_custom_onnx_config(export_source)
    print(f"  optimum main_export(dynamo=True) -> {out_dir}", flush=True)
    main_export(
        model_name_or_path=str(export_source),
        output=str(out_dir),
        task="token-classification",
        trust_remote_code=True,
        device="cpu",
        custom_onnx_configs={"model": onnx_config},
        dynamo=True,
        do_validation=False,
    )
    _copy_metadata(input_model, out_dir)
    shutil.rmtree(export_source, ignore_errors=True)
    onnx_files = list(out_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"export produced no .onnx file in {out_dir}")
    return onnx_files[0]


def export_fp32(input_model: Path, out_dir: Path, max_length: int) -> None:
    """Export straight to fp32 ONNX, no precision conversion. This is the
    variant known to work end-to-end (see module docstring); fp16 and int8
    both hit separate bugs in their respective third-party conversion tools
    when applied to this model's MoE graph shapes."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    fp32_path = _export_fp32_onnx(input_model, out_dir, max_length)
    _copy_metadata(input_model, out_dir)
    assert fp32_path.parent == out_dir


def export_fp16(input_model: Path, out_dir: Path, max_length: int) -> None:
    """Export fp32 then convert to fp16 with onnxconverter-common.

    STATUS: BLOCKED. `onnxconverter_common.float16.convert_float_to_float16`
    produces a graph with a genuine dtype mismatch on at least one `MatMul`
    node (mixed float16/float32 inputs, not just stale declared-vs-actual
    metadata) for this model's MoE structure — a real bug in that library's
    node-conversion heuristics for this graph shape, not something fixable
    with the existing `keep_io_types`/`disable_shape_infer` flags. Confirmed
    by also stripping stale `value_info` entries (a separate, cosmetic issue
    caused by `disable_shape_infer=True`) — the underlying node-level dtype
    mismatch persists regardless.
    """
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
    """Export fp32 then dynamic-quantize int8 weights via onnxruntime.quantization.

    STATUS: BLOCKED. `quantize_dynamic` raises `[ShapeInferenceError]
    Inferred shape and existing shape differ in dimension 0: (640) vs (128)`
    — a separate bug from the fp16 path, in onnxruntime's own shape
    inference during quantization, most likely tripped by the per-expert
    weight tensors' extra leading dimension (128 experts) that this
    quantizer's shape propagation wasn't written to expect. Confirmed with
    `per_channel=False` too (rules out a per-channel-quantization-axis mixup
    with the expert dimension) — the error persists either way.
    """
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
            if variant == "fp32":
                export_fp32(input_model, variant_dir, max_length)
            elif variant == "fp16":
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
