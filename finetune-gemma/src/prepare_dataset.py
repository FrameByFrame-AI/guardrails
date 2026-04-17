#!/usr/bin/env python3
"""Prepare and save the filtered chat-text dataset used by training."""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor

from training_data import (
    get_text_tokenizer,
    parse_type_caps,
    prepare_text_dataset,
    save_prepared_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Qwen training dataset")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--train-data", default="/data/guardrail_train.jsonl")
    parser.add_argument("--output-dir", default="/data/guardrail_train_prepared_gemma4_e2b")
    parser.add_argument("--chat-template", default="model_default")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--think-mode", choices=("required", "off", "mixed"), default="mixed")
    parser.add_argument("--think-ratio", type=float, default=0.2,
                        help="Fraction of records with thinking traces (only for --think-mode mixed)")
    parser.add_argument(
        "--type-cap",
        action="append",
        default=[],
        help="Optional per-type cap in the form TYPE=COUNT. May be specified multiple times.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_processor_loading_kwargs(model_name, local_files_only_flag):
    model_path = Path(model_name)
    is_local_dir = model_path.is_dir()
    return {
        "pretrained_model_name_or_path": model_name,
        "local_files_only": local_files_only_flag or is_local_dir,
    }, is_local_dir


def main():
    args = parse_args()
    train_path = Path(args.train_data)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    load_kwargs, is_local_dir = resolve_processor_loading_kwargs(
        args.model, args.local_files_only
    )
    print("Prepare configuration")
    print(f"  model: {args.model}")
    print(f"  local model path: {is_local_dir}")
    print(f"  local files only: {load_kwargs['local_files_only']}")
    print(f"  input data: {args.train_data}")
    print(f"  output dir: {args.output_dir}")
    print(f"  max seq length: {args.max_seq_length}")
    print(f"  think mode: {args.think_mode}")
    if args.type_cap:
        print(f"  type caps: {', '.join(args.type_cap)}")
    print("  chat template: model default")
    if args.chat_template != "qwen3-thinking":
        print(
            "  requested chat template is ignored during prepare step; "
            "the model's built-in template is used."
        )

    processor = AutoProcessor.from_pretrained(**load_kwargs)
    chat_processor = processor
    text_tokenizer = get_text_tokenizer(chat_processor)

    dataset = prepare_text_dataset(
        str(train_path),
        chat_processor,
        text_tokenizer,
        args.max_seq_length,
        args.max_train_samples,
        think_mode=args.think_mode,
        think_ratio=args.think_ratio,
        type_caps=parse_type_caps(args.type_cap),
    )
    save_prepared_dataset(dataset, args.output_dir)

    metadata = {
        "model": args.model,
        "chat_template": args.chat_template,
        "think_mode": args.think_mode,
        "type_cap": args.type_cap,
        "max_seq_length": args.max_seq_length,
        "max_train_samples": args.max_train_samples,
        "rows": len(dataset),
    }
    metadata_path = Path(args.output_dir) / "prep_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
