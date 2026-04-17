#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.

Usage:
    python merge_and_push.py --adapter /models/guardrail-qwen3.5-0.8b/checkpoint-500
    python merge_and_push.py --adapter /models/guardrail-qwen3.5-0.8b --output /models/guardrail-qwen3.5-0.8b-merged
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", default="", help="Output path (default: adapter path + -merged)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    output = args.output or args.adapter.rstrip("/") + "-merged"

    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    print(f"Saving merged 16-bit model to: {output}")
    Path(output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(output, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(output)
    print("Done.")


if __name__ == "__main__":
    main()
