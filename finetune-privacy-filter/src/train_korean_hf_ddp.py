#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

FALLBACK_SPAN_LABEL_MAP: dict[str, str] = {
    "personal_handle": "private_person",
    "ip_address": "private_url",
}


@dataclass(frozen=True)
class SpanRecord:
    category: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Hugging Face privacy-filter checkpoint on Korean PII."
    )
    parser.add_argument("--train-dataset", required=True, action="append",
                        help="Path to training JSONL. Pass multiple times to concat.")
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--test-dataset")
    parser.add_argument("--label-space-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many evals with no eval_span_f1 improvement. 0 disables.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=64)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--use-lora", action="store_true",
                        help="Apply LoRA to attention; freeze MoE experts/FFN; train classifier head fully.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated module name suffixes to apply LoRA to.")
    return parser.parse_args()


def _is_primary_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _print_primary(message: str) -> None:
    if _is_primary_process():
        print(message, flush=True)


def _prepare_output_dir(output_dir: Path, *, overwrite_output: bool) -> None:
    run_id = os.environ.get("PF_RUN_ID", "default")
    ready_file = output_dir / f".setup_complete_{run_id}"

    if _is_primary_process():
        if overwrite_output and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ready_file.write_text("ready\n", encoding="utf-8")
        return

    wait_s = 180.0
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if ready_file.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            return
        time.sleep(0.25)
    raise TimeoutError(
        f"Timed out waiting for primary rank to prepare output directory {output_dir}"
    )


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(
    path: str | os.PathLike[str], *, limit: int | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _load_span_class_names(path: str | os.PathLike[str]) -> list[str]:
    payload = _load_json(path)
    span_class_names = payload.get("span_class_names")
    if not isinstance(span_class_names, list) or not span_class_names:
        raise ValueError("label_space.json missing non-empty span_class_names")
    if span_class_names[0] != "O":
        raise ValueError("label_space.json must start span_class_names with 'O'")
    return [str(name) for name in span_class_names]


def _build_token_labels(span_class_names: list[str]) -> list[str]:
    token_labels = ["O"]
    for span_label in span_class_names:
        if span_label == "O":
            continue
        token_labels.extend(
            [
                f"B-{span_label}",
                f"I-{span_label}",
                f"E-{span_label}",
                f"S-{span_label}",
            ]
        )
    return token_labels


def _normalize_id2label(mapping: dict[Any, Any]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_key, raw_value in mapping.items():
        normalized[int(raw_key)] = str(raw_value)
    return normalized


def _extract_spans(record: dict[str, Any]) -> list[SpanRecord]:
    label_items = record.get("label")
    if isinstance(label_items, list):
        spans: list[SpanRecord] = []
        for item in label_items:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid label item: {item!r}")
            spans.append(
                SpanRecord(
                    category=str(item["category"]),
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
            )
        return spans

    spans_field = record.get("spans")
    if isinstance(spans_field, dict):
        spans = []
        for raw_category, offsets in spans_field.items():
            category = str(raw_category).split(": ", 1)[0]
            for start, end in offsets:
                spans.append(
                    SpanRecord(category=category, start=int(start), end=int(end))
                )
        return spans

    return []


def _project_spans_to_token_labels(
    *,
    text: str,
    spans: list[SpanRecord],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offset_mapping = [tuple(pair) for pair in encoded["offset_mapping"]]
    labels = [label_to_id["O"]] * len(input_ids)

    stats = {
        "token_count": len(input_ids),
        "span_count": len(spans),
        "spans_without_token_overlap": 0,
        "truncated_examples": 0,
    }

    for span in sorted(spans, key=lambda item: (item.start, item.end, item.category)):
        category = span.category
        if category not in {
            token_label.split("-", 1)[1]
            for token_label in label_to_id
            if token_label != "O"
        }:
            raise ValueError(f"Unsupported span label {category!r}")
        covered = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offset_mapping)
            if tok_end > tok_start and tok_end > span.start and tok_start < span.end
        ]
        if not covered:
            stats["spans_without_token_overlap"] += 1
            continue
        if any(labels[idx] != label_to_id["O"] for idx in covered):
            raise ValueError(
                "Overlapping token assignments detected while projecting spans for "
                f"text={text!r} span={span!r}"
            )
        if len(covered) == 1:
            labels[covered[0]] = label_to_id[f"S-{category}"]
        else:
            labels[covered[0]] = label_to_id[f"B-{category}"]
            for idx in covered[1:-1]:
                labels[idx] = label_to_id[f"I-{category}"]
            labels[covered[-1]] = label_to_id[f"E-{category}"]

    if encoded.get("overflow_to_sample_mapping"):
        stats["truncated_examples"] = 1

    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return features, stats


class JsonlTokenClassificationDataset(Dataset):
    def __init__(
        self,
        *,
        path: str | os.PathLike[str] | list[str | os.PathLike[str]],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
        max_examples: int | None,
        split_name: str,
    ) -> None:
        super().__init__()
        self.examples: list[dict[str, list[int]]] = []
        self.stats = {
            "split": split_name,
            "records": 0,
            "tokens": 0,
            "spans": 0,
            "spans_without_token_overlap": 0,
            "truncated_examples": 0,
            "max_tokens": 0,
            "records_per_path": {},
        }

        paths = path if isinstance(path, (list, tuple)) else [path]
        records: list[dict[str, Any]] = []
        for p in paths:
            loaded = _load_jsonl(p, limit=max_examples)
            self.stats["records_per_path"][str(p)] = len(loaded)
            records.extend(loaded)
            if max_examples is not None and len(records) >= max_examples:
                records = records[:max_examples]
                break
        for record in records:
            text = str(record["text"])
            spans = _extract_spans(record)
            features, local_stats = _project_spans_to_token_labels(
                text=text,
                spans=spans,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=max_length,
            )
            self.examples.append(features)
            self.stats["records"] += 1
            self.stats["tokens"] += local_stats["token_count"]
            self.stats["spans"] += local_stats["span_count"]
            self.stats["spans_without_token_overlap"] += local_stats[
                "spans_without_token_overlap"
            ]
            self.stats["truncated_examples"] += local_stats["truncated_examples"]
            self.stats["max_tokens"] = max(
                self.stats["max_tokens"], local_stats["token_count"]
            )

        if not self.examples:
            raise ValueError(f"No examples loaded for split {split_name!r} from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def _decode_bioes_spans(label_ids: list[int], id_to_label: dict[int, str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    active_label: str | None = None
    active_start: int | None = None

    for token_idx, label_id in enumerate(label_ids):
        label_name = id_to_label[int(label_id)]
        if label_name == "O":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
                active_label = None
                active_start = None
            continue

        prefix, category = label_name.split("-", 1)
        if prefix == "S":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
            spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
            continue
        if prefix == "B":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, token_idx))
            active_label = category
            active_start = token_idx
            continue
        if prefix == "I":
            if active_label != category or active_start is None:
                active_label = category
                active_start = token_idx
            continue
        if prefix == "E":
            if active_label == category and active_start is not None:
                spans.add((category, active_start, token_idx + 1))
            else:
                spans.add((category, token_idx, token_idx + 1))
            active_label = None
            active_start = None
            continue
        raise ValueError(f"Unsupported BIOES label {label_name!r}")

    if active_label is not None and active_start is not None:
        spans.add((active_label, active_start, len(label_ids)))
    return spans


def _build_metrics_fn(id_to_label: dict[int, str]):
    categories = sorted(
        {
            token_label.split("-", 1)[1]
            for token_label in id_to_label.values()
            if token_label != "O"
        }
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.asarray(logits).argmax(axis=-1)
        labels = np.asarray(label_ids)

        token_correct = 0
        token_total = 0
        gold_total = 0
        pred_total = 0
        true_positive = 0
        class_tp = {category: 0 for category in categories}
        class_pred_total = {category: 0 for category in categories}
        class_gold_total = {category: 0 for category in categories}

        for pred_seq, gold_seq in zip(predictions, labels, strict=True):
            keep_mask = gold_seq != -100
            pred_trimmed = pred_seq[keep_mask].astype(np.int64).tolist()
            gold_trimmed = gold_seq[keep_mask].astype(np.int64).tolist()
            token_total += len(gold_trimmed)
            token_correct += sum(
                int(pred == gold)
                for pred, gold in zip(pred_trimmed, gold_trimmed, strict=True)
            )

            pred_spans = _decode_bioes_spans(pred_trimmed, id_to_label)
            gold_spans = _decode_bioes_spans(gold_trimmed, id_to_label)
            pred_total += len(pred_spans)
            gold_total += len(gold_spans)
            true_positive += len(pred_spans & gold_spans)
            for category in categories:
                pred_for_category = {
                    span for span in pred_spans if span[0] == category
                }
                gold_for_category = {
                    span for span in gold_spans if span[0] == category
                }
                class_pred_total[category] += len(pred_for_category)
                class_gold_total[category] += len(gold_for_category)
                class_tp[category] += len(pred_for_category & gold_for_category)

        precision = true_positive / pred_total if pred_total else 0.0
        recall = true_positive / gold_total if gold_total else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        token_accuracy = token_correct / token_total if token_total else 0.0
        metrics: dict[str, float] = {
            "token_accuracy": token_accuracy,
            "span_precision": precision,
            "span_recall": recall,
            "span_f1": f1,
            "gold_spans": float(gold_total),
            "pred_spans": float(pred_total),
        }
        for category in categories:
            pred_count = class_pred_total[category]
            gold_count = class_gold_total[category]
            tp_count = class_tp[category]
            class_precision = tp_count / pred_count if pred_count else 0.0
            class_recall = tp_count / gold_count if gold_count else 0.0
            class_f1 = (
                2.0 * class_precision * class_recall / (class_precision + class_recall)
                if class_precision + class_recall
                else 0.0
            )
            metrics[f"class_{category}_precision"] = class_precision
            metrics[f"class_{category}_recall"] = class_recall
            metrics[f"class_{category}_f1"] = class_f1
            metrics[f"class_{category}_gold_spans"] = float(gold_count)
            metrics[f"class_{category}_pred_spans"] = float(pred_count)
        return metrics

    return compute_metrics


def _copy_classifier_rows(
    *,
    source_model: Any,
    target_model: Any,
    target_token_labels: list[str],
) -> dict[str, int]:
    if not hasattr(source_model, "score") or not hasattr(target_model, "score"):
        raise AttributeError("Expected both source and target models to expose .score")

    source_id2label = _normalize_id2label(source_model.config.id2label)
    source_label_to_id = {label: idx for idx, label in source_id2label.items()}
    copied_exact = 0
    copied_fallback = 0
    initialized_random = 0

    with torch.no_grad():
        for target_idx, target_label in enumerate(target_token_labels):
            source_idx = source_label_to_id.get(target_label)
            if source_idx is not None:
                target_model.score.weight[target_idx].copy_(
                    source_model.score.weight[source_idx]
                )
                target_model.score.bias[target_idx].copy_(
                    source_model.score.bias[source_idx]
                )
                copied_exact += 1
                continue

            if target_label == "O":
                initialized_random += 1
                continue

            prefix, category = target_label.split("-", 1)
            fallback_category = FALLBACK_SPAN_LABEL_MAP.get(category)
            if fallback_category is None:
                initialized_random += 1
                continue
            fallback_label = f"{prefix}-{fallback_category}"
            fallback_idx = source_label_to_id.get(fallback_label)
            if fallback_idx is None:
                initialized_random += 1
                continue
            target_model.score.weight[target_idx].copy_(
                source_model.score.weight[fallback_idx]
            )
            target_model.score.bias[target_idx].copy_(
                source_model.score.bias[fallback_idx]
            )
            copied_fallback += 1

    return {
        "exact_rows_copied": copied_exact,
        "fallback_rows_copied": copied_fallback,
        "random_rows_kept": initialized_random,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _prepare_output_dir(output_dir, overwrite_output=args.overwrite_output)

    set_seed(args.seed)
    span_class_names = _load_span_class_names(args.label_space_json)
    token_labels = _build_token_labels(span_class_names)
    label_to_id = {label: idx for idx, label in enumerate(token_labels)}
    id_to_label = {idx: label for idx, label in enumerate(token_labels)}

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    train_dataset = JsonlTokenClassificationDataset(
        path=list(args.train_dataset),
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
        max_examples=args.max_train_examples,
        split_name="train",
    )
    validation_dataset = JsonlTokenClassificationDataset(
        path=args.validation_dataset,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
        max_examples=args.max_validation_examples,
        split_name="validation",
    )
    test_dataset = None
    if args.test_dataset:
        test_dataset = JsonlTokenClassificationDataset(
            path=args.test_dataset,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_length=args.max_length,
            max_examples=args.max_test_examples,
            split_name="test",
        )

    _print_primary(
        "training plan: "
        f"train_records={train_dataset.stats['records']} "
        f"validation_records={validation_dataset.stats['records']} "
        f"test_records={0 if test_dataset is None else test_dataset.stats['records']} "
        f"max_length={args.max_length} "
        f"train_max_tokens={train_dataset.stats['max_tokens']} "
        f"validation_max_tokens={validation_dataset.stats['max_tokens']} "
        f"per_device_train_batch_size={args.per_device_train_batch_size} "
        f"per_device_eval_batch_size={args.per_device_eval_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}"
    )

    base_model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    config.num_labels = len(token_labels)
    config.id2label = id_to_label
    config.label2id = label_to_id
    config.use_cache = False

    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    classifier_remap_stats = _copy_classifier_rows(
        source_model=base_model,
        target_model=model,
        target_token_labels=token_labels,
    )
    del base_model

    lora_stats: dict[str, Any] | None = None
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["score"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        lora_stats = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        }
        _print_primary(f"LoRA enabled: trainable={trainable:,} / total={total:,} "
                       f"({lora_stats['trainable_pct']}%)")

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=args.save_total_limit,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        report_to=[],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_span_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=_build_metrics_fn(id_to_label),
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.use_lora:
        merged = trainer.model.merge_and_unload()
        trainer.model = merged
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test") if test_dataset is not None else None

    summary_payload = {
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "label_space_json": args.label_space_json,
        "token_labels": token_labels,
        "classifier_remap": classifier_remap_stats,
        "lora": lora_stats,
        "train_dataset": train_dataset.stats,
        "validation_dataset": validation_dataset.stats,
        "test_dataset": None if test_dataset is None else test_dataset.stats,
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    if _is_primary_process():
        shutil.copy2(args.label_space_json, output_dir / "label_space.json")
        with open(output_dir / "log_history.json", "w", encoding="utf-8") as handle:
            json.dump(trainer.state.log_history, handle, ensure_ascii=False, indent=2, default=str)
        with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        _print_primary(
            "classifier head remap: "
            f"exact={classifier_remap_stats['exact_rows_copied']} "
            f"fallback={classifier_remap_stats['fallback_rows_copied']} "
            f"random={classifier_remap_stats['random_rows_kept']}"
        )
        _print_primary(
            "validation metrics: "
            + " ".join(
                f"{key}={value:.4f}"
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float))
            )
        )
        if test_metrics is not None:
            _print_primary(
                "test metrics: "
                + " ".join(
                    f"{key}={value:.4f}"
                    for key, value in test_metrics.items()
                    if isinstance(value, (int, float))
                )
            )


if __name__ == "__main__":
    main()
