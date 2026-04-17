#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and optionally push to HuggingFace Hub.

Usage:
    # Just merge and save locally
    python merge_and_push.py --adapter /models/guardrail-gemma4-e2b --output /models/guardrail-gemma4-e2b-merged

    # Merge and push to HuggingFace
    python merge_and_push.py --adapter /models/guardrail-gemma4-e2b --output /models/guardrail-gemma4-e2b-merged --push --repo accuknox/guardrail-gemma4-e2b
"""

import argparse
from pathlib import Path

from unsloth import FastModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and optionally push to HF Hub")
    parser.add_argument("--adapter", default="/models/guardrail-gemma4-e2b",
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--output", default="/models/guardrail-gemma4-e2b-merged",
                        help="Output directory for merged model")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--repo", type=str, default="", help="HF repo id (e.g. accuknox/guardrail-gemma4-e2b)")
    parser.add_argument("--token", type=str, default="", help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"Loading adapter from: {args.adapter}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # Save merged 16-bit
    print(f"\nSaving merged 16-bit model to: {args.output}")
    model.save_pretrained_merged(
        args.output,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved.")

    # Push to HF Hub
    if args.push:
        if not args.repo:
            raise ValueError("--repo is required when using --push")
        token = args.token or None
        print(f"\nPushing to HuggingFace Hub: {args.repo}")
        model.push_to_hub_merged(
            args.repo,
            tokenizer,
            save_method="merged_16bit",
            token=token,
        )
        print(f"Pushed to https://huggingface.co/{args.repo}")

    print("\nDone!")


if __name__ == "__main__":
    main()
