#!/usr/bin/env python3
"""Fine-tune ModernBERT for programming language identification."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.dataset import CodeLanguageDataset
from code_language_id.snippets import SnippetConfig


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/llm_models/guardrail_code_models/modernbert-language-v1").expanduser()


def require_training_deps() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with:\n"
            "  python3 -m pip install --user -r code-language-id/requirements.txt"
        ) from exc

    return {
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def load_labels(data_root: Path) -> tuple[list[str], dict[int, str], dict[str, int]]:
    labels_path = data_root / "interim/reports/v1_languages.csv"
    labels = pd.read_csv(labels_path).sort_values("label_id")
    names = labels["canonical_language"].tolist()
    id2label = {int(row.label_id): row.canonical_language for row in labels.itertuples()}
    label2id = {label: idx for idx, label in id2label.items()}
    if len(names) != len(id2label):
        raise ValueError(f"Duplicate label ids in {labels_path}")
    return names, id2label, label2id


def limit_dataset(dataset: CodeLanguageDataset, limit: int | None) -> CodeLanguageDataset:
    if limit is not None:
        dataset.frame = dataset.frame.head(limit).copy()
    return dataset


@dataclass
class TokenizingCollator:
    tokenizer: Any
    torch: Any
    max_length: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = self.torch.tensor(
            [item["labels"] for item in batch],
            dtype=self.torch.long,
        )
        return encoded


def build_epoch_callback(base_class: Any, train_dataset: CodeLanguageDataset) -> Any:
    class SnippetEpochCallback(base_class):
        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            epoch = int(state.epoch or 0)
            train_dataset.set_epoch(epoch)

    return SnippetEpochCallback()


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    accuracy = float((predictions == labels).mean())

    num_labels = int(max(labels.max(), predictions.max())) + 1
    f1_scores = []
    for label_id in range(num_labels):
        true_positive = int(((predictions == label_id) & (labels == label_id)).sum())
        false_positive = int(((predictions == label_id) & (labels != label_id)).sum())
        false_negative = int(((predictions != label_id) & (labels == label_id)).sum())
        if true_positive == 0 and false_positive == 0 and false_negative == 0:
            continue
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


def make_training_arguments(training_arguments_cls: Any, args: argparse.Namespace) -> Any:
    kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_total_limit": args.save_total_limit,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
    }

    signature = inspect.signature(training_arguments_cls.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "logging_strategy" in signature.parameters:
        kwargs["logging_strategy"] = "steps"

    return training_arguments_cls(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--splits-subdir",
        default="processed/v1_splits",
        help="Path relative to data-root that contains train/val/test parquets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-chars", type=int, default=512)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--tokenizer-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop training if macro_f1 does not improve for this many evals. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute improvement in macro_f1 to reset the patience counter.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="HF LR scheduler type: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup.",
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--reference-compile",
        action="store_true",
        help="Keep ModernBERT's torch.compile reference path enabled.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation. Use eager by default; sdpa produced NaN loss on this stack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deps = require_training_deps()
    torch = deps["torch"]
    auto_tokenizer_cls = deps["AutoTokenizer"]
    auto_model_cls = deps["AutoModelForSequenceClassification"]
    early_stopping_cls = deps["EarlyStoppingCallback"]
    trainer_cls = deps["Trainer"]
    trainer_callback_cls = deps["TrainerCallback"]
    training_arguments_cls = deps["TrainingArguments"]

    _, id2label, label2id = load_labels(args.data_root)
    snippet_config = SnippetConfig(
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        seed=args.seed,
    )

    splits_dir = args.data_root / args.splits_subdir
    train_dataset = CodeLanguageDataset(
        splits_dir / "train.parquet",
        mode="train",
        snippet_config=snippet_config,
    )
    eval_dataset = CodeLanguageDataset(
        splits_dir / "val.parquet",
        mode="eval",
        snippet_config=snippet_config,
    )
    train_dataset = limit_dataset(train_dataset, args.max_train_samples)
    eval_dataset = limit_dataset(eval_dataset, args.max_eval_samples)

    tokenizer = auto_tokenizer_cls.from_pretrained(args.model_name)
    model_kwargs = {
        "num_labels": len(id2label),
        "id2label": id2label,
        "label2id": label2id,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = auto_model_cls.from_pretrained(args.model_name, **model_kwargs)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = args.reference_compile
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    training_args = make_training_arguments(training_arguments_cls, args)
    collator = TokenizingCollator(
        tokenizer=tokenizer,
        torch=torch,
        max_length=args.tokenizer_max_length,
    )

    metadata = {
        "model_name": args.model_name,
        "num_labels": len(id2label),
        "id2label": id2label,
        "label2id": label2id,
        "attn_implementation": args.attn_implementation,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "snippet_config": {
            "max_chars": snippet_config.max_chars,
            "min_chars": snippet_config.min_chars,
            "short_chars": snippet_config.short_chars,
            "train_strategies": list(snippet_config.train_strategies),
            "eval_strategy": snippet_config.eval_strategy,
            "seed": snippet_config.seed,
        },
    }
    with (args.output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    callbacks = [build_epoch_callback(trainer_callback_cls, train_dataset)]
    if args.early_stopping_patience > 0:
        callbacks.append(
            early_stopping_cls(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
