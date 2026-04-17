#!/usr/bin/env python3
"""
Train GLiNER2 Korean PII NER.

Loads GLiNER2 base (English DeBERTa), extends the tokenizer with top Korean
subword tokens from the training data, resizes encoder embeddings, then
fine-tunes on Korean PII data.

Usage (inside docker):
    python train_pii.py
    python train_pii.py --epochs 15 --batch-size 8
    python train_pii.py --korean-tokens 5000
    python train_pii.py --lora --resume-checkpoint /models/run/checkpoint-epoch-3
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from types import MethodType

import torch
from gliner2 import GLiNER2
from gliner2.training.lora import LoRAAdapterConfig
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from huggingface_hub import hf_hub_download
from transformers import AutoModel, PreTrainedTokenizerFast


KOREAN_TOKENIZER = "team-lucid/deberta-v3-base-korean"
DEFAULT_LORA_TARGETS = [
    "encoder",
    "span_rep",
    "classifier",
    "count_embed",
    "count_pred",
]


def has_hangul(text: str) -> bool:
    return any(
        "\uAC00" <= c <= "\uD7A3" or "\u3131" <= c <= "\u318E"
        for c in text
    )


def parse_lora_targets(raw: str) -> list[str]:
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if not targets:
        raise ValueError("LoRA targets cannot be empty when LoRA is enabled")
    return targets


def configure_gradient_checkpointing(model) -> None:
    encoder = model.encoder
    if not hasattr(encoder, "gradient_checkpointing_enable"):
        raise RuntimeError(
            f"Encoder {type(encoder).__name__} does not support gradient checkpointing"
        )

    encoder.gradient_checkpointing_enable()

    # Required by some HF encoders so gradients can flow when most base
    # weights are frozen (for example, LoRA training).
    if hasattr(encoder, "enable_input_require_grads"):
        encoder.enable_input_require_grads()

    print(f"Gradient checkpointing enabled on {type(encoder).__name__}")


def install_safe_embedding_extraction(processor) -> None:
    if getattr(processor, "_safe_extract_installed", False):
        return

    def safe_extract_embeddings_from_batch(self, token_embeddings, input_ids, batch):
        use_fast_path = (
            self.token_pooling == "first"
            and batch.text_word_indices is not None
            and batch.schema_special_indices is not None
        )
        if not use_fast_path:
            return self._extract_embeddings_loop(token_embeddings, input_ids, batch)

        try:
            for i, expected_schemas in enumerate(batch.schema_counts):
                if i >= len(batch.schema_special_indices):
                    raise IndexError(
                        f"missing schema_special_indices for sample {i}"
                    )
                if len(batch.schema_special_indices[i]) < expected_schemas:
                    raise IndexError(
                        "schema_special_indices shorter than schema_counts "
                        f"for sample {i}: "
                        f"{len(batch.schema_special_indices[i])} < {expected_schemas}"
                    )
            return self._extract_embeddings_fast(token_embeddings, batch)
        except IndexError as exc:
            if not getattr(self, "_safe_extract_warned", False):
                print(
                    "Fast schema embedding extraction failed; "
                    "falling back to safe loop path. "
                    f"Reason: {exc}"
                )
                self._safe_extract_warned = True
            return self._extract_embeddings_loop(token_embeddings, input_ids, batch)

    processor.extract_embeddings_from_batch = MethodType(
        safe_extract_embeddings_from_batch,
        processor,
    )
    processor._safe_extract_installed = True
    print("Safe schema embedding fallback installed")


def extend_with_korean_tokens(
    model,
    train_path: Path,
    top_k: int = 5000,
    sample_limit: int = 10000,
    init_from_korean: bool = True,
):
    """Add Korean tokenizer entries and optionally seed them from Korean embeddings."""
    print(f"\nExtending tokenizer with Korean tokens...")

    kr_tok_path = hf_hub_download(KOREAN_TOKENIZER, "tokenizer.json")
    kr_tok = PreTrainedTokenizerFast(tokenizer_file=kr_tok_path)
    kr_tok.pad_token = "[PAD]"
    kr_tok.cls_token = "[CLS]"
    kr_tok.sep_token = "[SEP]"
    kr_tok.unk_token = "[UNK]"
    kr_tok.mask_token = "[MASK]"
    kr_encoder = None
    if init_from_korean:
        kr_encoder = AutoModel.from_pretrained(KOREAN_TOKENIZER)
        kr_encoder.eval()

    target_tokenizer = model.processor.tokenizer
    target_vocab = set(target_tokenizer.get_vocab().keys())

    # Score Korean tokens by frequency in training data
    token_freq = Counter()
    with train_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_limit > 0 and i >= sample_limit:
                break
            r = json.loads(line)
            tokens = kr_tok.tokenize(r.get("input", ""))
            for t in tokens:
                if t not in target_vocab and has_hangul(t):
                    token_freq[t] += 1

    # Take top-K most frequent
    new_tokens = [t for t, _ in token_freq.most_common(top_k)]
    original_vocab_size = len(target_tokenizer)
    num_added = target_tokenizer.add_tokens(new_tokens)
    model.encoder.resize_token_embeddings(len(target_tokenizer))

    initialized = 0
    if init_from_korean and num_added > 0:
        target_embeddings = model.encoder.embeddings.word_embeddings.weight
        source_embeddings = kr_encoder.embeddings.word_embeddings.weight
        with torch.no_grad():
            for token in new_tokens:
                target_id = target_tokenizer.convert_tokens_to_ids(token)
                if target_id < original_vocab_size:
                    continue
                source_id = kr_tok.convert_tokens_to_ids(token)
                if source_id == kr_tok.unk_token_id:
                    continue
                target_embeddings[target_id].copy_(
                    source_embeddings[source_id].to(
                        device=target_embeddings.device,
                        dtype=target_embeddings.dtype,
                    )
                )
                initialized += 1

    print(f"  Korean tokens found: {len(token_freq)}")
    print(f"  Added: {num_added}, new vocab: {len(target_tokenizer)}")
    if init_from_korean:
        print(f"  Embeddings initialized from Korean encoder: {initialized}")
    print(f"  Embedding shape: {model.encoder.embeddings.word_embeddings.weight.shape}")
    if new_tokens:
        print(f"  Sample new tokens: {new_tokens[:10]}")

    # Test
    test = "김민수의 전화번호는 010-1234-5678입니다"
    tokens = target_tokenizer.tokenize(test)
    print(f"  Test: '{test}' -> {tokens} ({len(tokens)} tokens)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GLiNER2 Korean PII NER")
    parser.add_argument("--model", type=str, default="fastino/gliner2-base-v1")
    parser.add_argument("--train-data", type=str, default="/data/pii_gliner2_train.jsonl")
    parser.add_argument("--output-dir", type=str, default="/models/gliner2-korean-pii")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=5,
                        help="Gradient accumulation steps")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--korean-tokens", type=int, default=5000,
                        help="Number of Korean tokens to add (0=skip)")
    parser.add_argument("--token-sample-limit", type=int, default=10000,
                        help="Number of training records to scan when mining Korean tokens")
    parser.add_argument("--no-korean-embedding-init", action="store_true",
                        help="Do not seed new token embeddings from the Korean encoder")
    parser.add_argument("--max-train-samples", type=int, default=-1,
                        help="Cap train samples for smoke tests (-1 uses all)")
    parser.add_argument("--max-eval-samples", type=int, default=-1,
                        help="Cap eval samples for smoke tests (-1 uses all)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable encoder gradient checkpointing")
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA adapter training")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout")
    parser.add_argument(
        "--lora-targets",
        type=str,
        default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated GLiNER2 module groups for LoRA",
    )
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--fp16", action="store_true",
                                 help="Enable fp16 mixed precision")
    precision_group.add_argument("--bf16", action="store_true",
                                 help="Enable bf16 mixed precision")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help=(
            "Load model or LoRA adapter weights from this directory before "
            "training. Adapter dirs contain adapter_config.json; full saves "
            "look like output/best or checkpoint-epoch-N. Restores weights "
            "only (optimizer and LR schedule still start over). Use the same "
            "--model and --korean-tokens as the run that wrote the checkpoint."
        ),
    )
    args = parser.parse_args()

    train_path = Path(args.train_data)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    with train_path.open() as f:
        total = sum(1 for _ in f)
    print(f"Training data: {train_path} ({total} records)")

    # Load GLiNER2
    print(f"Loading model: {args.model}")
    model = GLiNER2.from_pretrained(args.model)
    print(f"  Encoder vocab: {model.encoder.config.vocab_size}")
    install_safe_embedding_extraction(model.processor)

    # Extend tokenizer with Korean tokens
    if args.korean_tokens > 0:
        model = extend_with_korean_tokens(
            model,
            train_path,
            top_k=args.korean_tokens,
            sample_limit=args.token_sample_limit,
            init_from_korean=not args.no_korean_embedding_init,
        )

    if args.gradient_checkpointing:
        configure_gradient_checkpointing(model)

    has_eval = args.val_ratio > 0

    # Training config
    config = TrainingConfig(
        output_dir=args.output_dir,
        experiment_name="gliner2-korean-pii",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        scheduler_type="cosine",
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_best=has_eval,
        early_stopping=has_eval,
        early_stopping_patience=3,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        use_lora=args.lora,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_lora_targets(args.lora_targets),
        report_to_wandb=bool(args.wandb_project),
        wandb_project=args.wandb_project or None,
    )

    precision = "fp32"
    if args.fp16:
        precision = "fp16"
    elif args.bf16:
        precision = "bf16"
    print(f"Training precision: {precision}")
    if not has_eval:
        print("Validation disabled: save_best and early_stopping are off.")
    if args.lora:
        print(
            "LoRA enabled: "
            f"rank={args.lora_rank}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, "
            f"targets={parse_lora_targets(args.lora_targets)}"
        )

    trainer = GLiNER2Trainer(model, config)

    resume = (args.resume_checkpoint or "").strip()
    if resume:
        resume_path = Path(resume)
        if not resume_path.is_dir():
            raise FileNotFoundError(
                f"--resume-checkpoint must be a directory: {resume_path}"
            )
        if LoRAAdapterConfig.is_adapter_path(resume_path) and not args.lora:
            raise ValueError(
                "Adapter checkpoint (adapter_config.json present) requires --lora"
            )
        print(f"Resuming weights from {resume_path} (optimizer state is not restored)")
        trainer.load_checkpoint(str(resume_path))

    if args.val_ratio > 0:
        import random
        lines = train_path.read_text(encoding="utf-8").strip().splitlines()
        random.Random(42).shuffle(lines)
        split_idx = int(len(lines) * (1.0 - args.val_ratio))

        val_path = train_path.with_suffix(".val.jsonl")
        train_split_path = train_path.with_suffix(".train.jsonl")
        train_split_path.write_text("\n".join(lines[:split_idx]) + "\n", encoding="utf-8")
        val_path.write_text("\n".join(lines[split_idx:]) + "\n", encoding="utf-8")

        print(f"Train: {split_idx} | Val: {len(lines) - split_idx}")
        results = trainer.train(
            train_data=str(train_split_path),
            eval_data=str(val_path),
        )
    else:
        results = trainer.train(train_data=str(train_path))

    print(f"\nTraining complete!")
    if results:
        # Print summary only, not full step-by-step history
        summary = {k: v for k, v in results.items() if k not in ("history", "metrics_history", "train_history")}
        print(f"Summary: {json.dumps(summary, indent=2, default=str)}")

    model.save_pretrained(args.output_dir)
    model.processor.tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
