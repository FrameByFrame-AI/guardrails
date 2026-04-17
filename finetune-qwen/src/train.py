#!/usr/bin/env python3
"""Fine-tune a Qwen guardrail model with Unsloth LoRA/QLoRA."""

import argparse
import os
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported
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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for guardrail classification")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_BASE_MODEL", "/models/unsloth/Qwen3.5-0.8B"),
    )
    parser.add_argument("--train-data", default="/data/guardrail_train.jsonl")
    parser.add_argument("--prepared-data", default="/data/guardrail_train_prepared")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("QWEN_OUTPUT_DIR", "/models/guardrail-qwen3.5-0.8b"),
    )
    parser.add_argument("--chat-template", default="qwen3-thinking")
    parser.add_argument("--think-mode", choices=("required", "off", "mixed"), default="mixed")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", nargs="+", default=DEFAULT_LORA_TARGETS)
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("off", "true", "unsloth"),
        default="unsloth",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-on-the-fly", action="store_true")
    parser.add_argument("--save-merged-16bit", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir")
    parser.add_argument(
        "--merged-output-dir",
        default=os.environ.get("QWEN_VLLM_MODEL", ""),
        help="Optional path for the merged full model used by vLLM / Hugging Face upload.",
    )
    return parser.parse_args()


def resolve_gradient_checkpointing(mode: str):
    if mode == "off":
        return False
    if mode == "true":
        return True
    return "unsloth"


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
        print(f"  bf16: {bf16_supported}")

    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        **model_load_kwargs,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    print(f"\nConfiguring LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_targets,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=resolve_gradient_checkpointing(args.gradient_checkpointing),
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    chat_processor = get_chat_template(tokenizer, chat_template=args.chat_template)
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
            warmup_ratio=args.warmup_ratio,
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

    # Mask system and user tokens. Training still covers the full assistant turn
    # beginning either at <think> or directly at the JSON verdict.
    response_part = "<|im_start|>assistant\n<think>"
    if args.think_mode == "off":
        response_part = "<|im_start|>assistant\n"
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part=response_part,
    )

    print(f"\nStarting training...")
    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)
    print(f"Training complete. Final loss: {trainer_stats.training_loss:.6f}")

    print(f"\nSaving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    chat_processor.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        merged_dir = args.merged_output_dir or (args.output_dir + "-merged-16bit")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving merged 16-bit model to {merged_dir}")
        model.save_pretrained_merged(merged_dir, text_tokenizer, save_method="merged_16bit")
        chat_processor.save_pretrained(merged_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
