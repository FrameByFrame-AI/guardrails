#!/usr/bin/env python3
"""Phase 5: export the (optionally LoRA-merged) Granite encoder to ONNX.

INT8 quantization is required to hit the <30 ms CPU latency budget. Topic
embeddings are computed offline and shipped alongside the encoder for the
default topic list; runtime requests still re-encode novel topics through the
ONNX session and cache them in-process.

    docker compose run --rm export-bantopics-onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, type=Path,
                   help="Sentence-Transformers checkpoint dir or HF id.")
    p.add_argument("--lora-adapter", type=Path, default=None,
                   help="Optional LoRA adapter to merge before export.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--quantize", choices=["none", "int8", "fp16"], default="int8")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(args.model))
    if args.lora_adapter:
        from peft import PeftModel

        base = encoder._first_module().auto_model
        merged = PeftModel.from_pretrained(base, str(args.lora_adapter)).merge_and_unload()
        encoder._first_module().auto_model = merged

    args.output.mkdir(parents=True, exist_ok=True)
    merged_dir = args.output / "merged"
    encoder.save(str(merged_dir))

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    onnx_dir = args.output / "onnx"
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        str(merged_dir), export=True, opset=args.opset,
    )
    ort_model.save_pretrained(str(onnx_dir))

    if args.quantize == "int8":
        quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=str(onnx_dir / "int8"), quantization_config=qconfig)

    (args.output / "export_meta.json").write_text(json.dumps({
        "base_model": str(args.model),
        "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
        "quantize": args.quantize,
        "opset": args.opset,
    }, indent=2))
    print(f"exported to {args.output}")


if __name__ == "__main__":
    main()
