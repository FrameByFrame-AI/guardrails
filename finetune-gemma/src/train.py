#!/usr/bin/env python3
"""Fine-tune a Gemma 4 guardrail model with Unsloth LoRA."""

import argparse
import os
from pathlib import Path

from unsloth import FastModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

from training_data import get_text_tokenizer, load_prepared_dataset, prepare_text_dataset


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for guardrail classification")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--train-data", default="/data/guardrail_train.jsonl")
    parser.add_argument("--prepared-data", default="/data/guardrail_train_prepared_gemma4_e2b")
    parser.add_argument("--output-dir", default="/models/guardrail-gemma4-e2b")
    parser.add_argument("--chat-template", default="model_default")
    parser.add_argument(
        "--model-family",
        default="gemma-4",
        choices=("gemma-4", "qwen3", "generic"),
        help="Model architecture family — controls the response-mask markers used "
             "during training. Defaults to gemma-4 since this is finetune-gemma/. "
             "Do NOT rely on --model path substring matching.",
    )
    parser.add_argument("--think-mode", choices=("required", "off", "mixed"), default="mixed")
    parser.add_argument("--max-seq-length", type=int, default=2048)  # sufficient for guardrail JSON output
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", nargs="+", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--load-precision", choices=("4bit", "16bit"), default="4bit")
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("off", "true", "unsloth"),
        default="unsloth",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-on-the-fly", action="store_true")
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument("--init-lora-from", default="",
                        help="Path to an existing LoRA adapter directory; loaded as initial "
                             "adapter weights instead of random init. Use when continuing "
                             "training on new data from a prior run.")
    return parser.parse_args()


def resolve_gradient_checkpointing(mode: str):
    if mode == "off":
        return False
    if mode == "true":
        return True
    return "unsloth"


def resolve_chat_processor(tokenizer, chat_template: str):
    if chat_template in ("model_default", "default", "auto"):
        return tokenizer
    return get_chat_template(tokenizer, chat_template=chat_template)


def resolve_response_only_markers(model_family: str, chat_template: str, think_mode: str):
    """Return (instruction_part, response_part) for train_on_responses_only masking.

    Chat template overrides take precedence; otherwise we select markers based
    on the model family. Missing markers cause loss to be computed over the
    full conversation (user + assistant) which is ~10x the assistant-only loss.
    """
    if chat_template not in ("model_default", "default", "auto"):
        response_part = "<|im_start|>assistant\n<think>"
        if think_mode == "off":
            response_part = "<|im_start|>assistant\n"
        return "<|im_start|>user\n", response_part

    if model_family == "gemma-4":
        return "<|turn>user\n", "<|turn>model\n"
    if model_family == "qwen3":
        response_part = "<|im_start|>assistant\n<think>"
        if think_mode == "off":
            response_part = "<|im_start|>assistant\n"
        return "<|im_start|>user\n", response_part
    return None, None


def resolve_model_loading_kwargs(model_name: str, local_files_only_flag: bool):
    model_path = Path(model_name)
    is_local_dir = model_path.is_dir()
    return {
        "model_name": model_name,
        "local_files_only": local_files_only_flag or is_local_dir,
    }, is_local_dir


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    effective_batch = args.batch_size * args.grad_accum * world_size
    bf16_supported = is_bfloat16_supported()
    model_load_kwargs, is_local_dir = resolve_model_loading_kwargs(
        args.model, args.local_files_only
    )

    if local_rank == 0:
        print("\nRun configuration")
        print(f"  model: {args.model}")
        print(f"  local model path: {is_local_dir}")
        print(f"  local files only: {model_load_kwargs['local_files_only']}")
        print(f"  world size: {world_size}")
        print(f"  per-device batch: {args.batch_size}")
        print(f"  grad accumulation: {args.grad_accum}")
        print(f"  effective global batch: {effective_batch}")
        print(f"  max seq length: {args.max_seq_length}")
        print(f"  think mode: {args.think_mode}")
        print(f"  load precision: {args.load_precision}")
        print(f"  bf16: {bf16_supported}")

    print(f"\nLoading model: {args.model}")
    load_kwargs = {
        **model_load_kwargs,
        "max_seq_length": args.max_seq_length,
        "dtype": None,  # auto detection
        "full_finetuning": False,
    }
    if args.load_precision == "4bit":
        load_kwargs["load_in_4bit"] = True
    if world_size > 1:
        load_kwargs["device_map"] = "balanced"
    model, tokenizer = FastModel.from_pretrained(**load_kwargs)

    print(f"\nConfiguring LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
    )

    # Optionally initialize LoRA from a previous adapter (continue training).
    if args.init_lora_from:
        from peft.utils.save_and_load import set_peft_model_state_dict
        from safetensors.torch import load_file
        adapter_path = Path(args.init_lora_from) / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter weights not found: {adapter_path}")
        print(f"\nLoading initial LoRA weights from {adapter_path}")
        adapter_state = load_file(str(adapter_path))
        load_result = set_peft_model_state_dict(model, adapter_state)
        n_missing = len(getattr(load_result, "missing_keys", []) or [])
        n_unexpected = len(getattr(load_result, "unexpected_keys", []) or [])
        n_loaded = len(adapter_state) - n_unexpected
        print(f"  Loaded: {n_loaded} tensors | missing: {n_missing} | unexpected: {n_unexpected}")
        if n_loaded == 0:
            raise RuntimeError("No adapter tensors loaded — check path / LoRA rank alignment")

    chat_processor = resolve_chat_processor(tokenizer, args.chat_template)
    text_tokenizer = get_text_tokenizer(chat_processor)

    print("\nLoading training dataset...")
    prepared_path = Path(args.prepared_data)
    if prepared_path.exists():
        dataset = load_prepared_dataset(str(prepared_path), args.max_train_samples)
    elif args.prepare_on_the_fly:
        train_path = Path(args.train_data)
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        dataset = prepare_text_dataset(
            str(train_path),
            chat_processor,
            text_tokenizer,
            args.max_seq_length,
            args.max_train_samples,
        )
    else:
        raise FileNotFoundError(
            f"Prepared dataset not found: {prepared_path}. Run prepare_dataset.py first "
            "or pass --prepare-on-the-fly."
        )

    print(f"\nConfiguring trainer...")
    print(f"  Per-device batch: {args.batch_size}")
    print(f"  Effective global batch: {effective_batch}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")

    sft_kwargs = {}
    if world_size > 1:
        sft_kwargs["ddp_find_unused_parameters"] = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=args.dataset_num_proc,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_strategy="steps",
            output_dir=args.output_dir,
            report_to="none",
            bf16=bf16_supported,
            fp16=not bf16_supported,
            max_length=args.max_seq_length,
            remove_unused_columns=False,
            **sft_kwargs,
        ),
    )

    instruction_part, response_part = resolve_response_only_markers(
        args.model_family, args.chat_template, args.think_mode
    )
    if not (instruction_part and response_part):
        raise RuntimeError(
            f"No response-only mask markers for model_family={args.model_family!r}. "
            "Pass --model-family explicitly. Training without masking would "
            "compute loss over the full conversation — see earlier bug."
        )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    print(f"\nStarting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Final loss: {trainer_stats.training_loss:.6f}")

    print(f"\nSaving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    chat_processor.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        merged_dir = args.output_dir + "-merged-16bit"
        print(f"Saving merged 16-bit model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, text_tokenizer, save_method="merged_16bit")
        chat_processor.save_pretrained(merged_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
