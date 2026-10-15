#!/usr/bin/env python3
"""Export the ByteHybrid lite checkpoint to ONNX.

Produces ``model.onnx`` next to the source checkpoint with a single dynamic
``batch`` axis on the ``byte_ids`` input and ``logits`` output. ``lang2idx.json``
is also copied alongside so downstream consumers don't need the .pt file.

After export, this script runs both the PyTorch and onnxruntime models on a
fixed set of random inputs and reports the max absolute / relative difference
in logits — a parity check, not a full eval.

    python scripts/export_byte_hybrid_onnx.py \\
        --checkpoint /models/guardrail_code_models/programming-language-identification-100plus-lite/model.pt \\
        --output     /models/guardrail_code_models/programming-language-identification-100plus-lite/onnx/model.onnx
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


def load_model(checkpoint_path: Path) -> tuple[ByteHybrid, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=ckpt["max_len"],
        **CONFIGS[ckpt["config"]],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(
    model: ByteHybrid,
    max_len: int,
    output_path: Path,
    opset: int,
    sample_batch: int = 4,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randint(0, 257, (sample_batch, max_len), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["byte_ids"],
        output_names=["logits"],
        dynamic_axes={"byte_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )


def verify_parity(
    model: ByteHybrid,
    onnx_path: Path,
    max_len: int,
    num_classes: int,
    num_samples: int = 8,
    seed: int = 0,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed; skipping parity check.", flush=True)
        return {"skipped": True}

    rng = np.random.default_rng(seed)
    byte_ids = rng.integers(0, 257, size=(num_samples, max_len), dtype=np.int64)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(byte_ids)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"byte_ids": byte_ids})[0]

    diff = np.abs(torch_out - onnx_out)
    max_abs = float(diff.max())
    max_rel = float((diff / np.clip(np.abs(torch_out), 1e-9, None)).max())
    torch_argmax = torch_out.argmax(axis=-1)
    onnx_argmax = onnx_out.argmax(axis=-1)
    argmax_match = float((torch_argmax == onnx_argmax).mean())

    print(
        f"parity: samples={num_samples} num_classes={num_classes} "
        f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} "
        f"argmax_match={argmax_match:.3f}",
        flush=True,
    )
    return {
        "samples": num_samples,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "argmax_match": argmax_match,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/models/guardrail_code_models/"
            "programming-language-identification-100plus-lite/model.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path. Default: <checkpoint dir>/onnx/model.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset. torch>=2.11 dynamo exporter emits ~opset 20-22 natively; "
             "down-converting to 17 fails on ops like Pad. Use 20 for broad runtime support.",
    )
    parser.add_argument("--sample-batch", type=int, default=4)
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--skip-parity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.checkpoint.parent / "onnx" / "model.onnx"

    model, ckpt = load_model(args.checkpoint)
    max_len = int(ckpt["max_len"])
    num_classes = int(ckpt["num_classes"])

    print(f"Exporting {args.checkpoint} -> {args.output} (opset={args.opset})", flush=True)
    export_onnx(model, max_len, args.output, args.opset, args.sample_batch)

    sidecar = args.output.parent / "lang2idx.json"
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(ckpt["lang2idx"], handle, indent=2, sort_keys=True)
    print(f"Wrote {sidecar}", flush=True)

    metadata_path = args.output.parent / "onnx_metadata.json"
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "config": ckpt["config"],
        "num_classes": num_classes,
        "max_len": max_len,
        "opset": args.opset,
    }
    if not args.skip_parity:
        metadata["parity"] = verify_parity(
            model, args.output, max_len, num_classes, args.parity_samples
        )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
