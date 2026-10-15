#!/usr/bin/env python3
"""Train ByteHybrid for programming-language identification (lite model).

This is the byte-level counterpart to ``train_modernbert_language.py``: same
data, same labels, same snippet pipeline — but a 2.35M-parameter byte-level
classifier instead of a 150M+ tokenizer-based transformer. The shipped
checkpoint is saved in CommonLingua's format so it loads with the upstream
``predict.py`` (or our ``byte_hybrid_predict.py``) without modification.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from code_language_id.byte_hybrid import CONFIGS, ByteHybrid
from code_language_id.dataset import CodeLanguageDataset
from code_language_id.snippets import SnippetConfig


DEFAULT_DATA_ROOT = Path("~/llm_models/guardrail_code_data").expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/llm_models/guardrail_code_models/programming-language-identification-100plus-lite"
).expanduser()


def load_labels(data_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    labels_path = data_root / "interim/reports/v1_languages.csv"
    labels = pd.read_csv(labels_path).sort_values("label_id")
    id2label = {int(row.label_id): row.canonical_language for row in labels.itertuples()}
    label2id = {label: idx for idx, label in id2label.items()}
    if len(label2id) != len(id2label):
        raise ValueError(f"Duplicate label ids in {labels_path}")
    return id2label, label2id


def encode_bytes(texts: list[str], max_len: int) -> torch.Tensor:
    """Encode UTF-8 bytes; pad with 256 (same convention as CommonLingua)."""
    out = np.full((len(texts), max_len), 256, dtype=np.int64)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        raw = t.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


class ByteCollator:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        byte_ids = encode_bytes([item["text"] for item in batch], self.max_len)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"byte_ids": byte_ids, "labels": labels}


def cosine_warmup(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: ByteHybrid,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        byte_ids = batch["byte_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(byte_ids)
        else:
            logits = model(byte_ids)
        preds = logits.float().argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for c in range(num_classes):
            pred_c = preds_np == c
            true_c = labels_np == c
            tp[c] += int(np.logical_and(pred_c, true_c).sum())
            fp[c] += int(np.logical_and(pred_c, ~true_c).sum())
            fn[c] += int(np.logical_and(~pred_c, true_c).sum())

    f1s: list[float] = []
    for c in range(num_classes):
        if tp[c] == 0 and fp[c] == 0 and fn[c] == 0:
            continue
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall = tp[c] / max(tp[c] + fn[c], 1)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))

    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "num_eval": total,
    }


def save_checkpoint(
    path: Path,
    model: ByteHybrid,
    num_classes: int,
    max_len: int,
    lang2idx: dict[str, int],
    config_name: str,
    dtype: torch.dtype | None = None,
) -> None:
    state = model.state_dict()
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    payload = {
        "model_state_dict": state,
        "num_classes": num_classes,
        "max_len": max_len,
        "lang2idx": lang2idx,
        "config": config_name,
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--splits-subdir",
        default="processed/v1_splits_clean",
        help="Path relative to data-root that contains train/val/test parquets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-name", default="base_ngram", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--max-len", type=int, default=1023, help="Max bytes per sample.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1023,
        help="Snippet character cap; should be >= max-len for ASCII-heavy code.",
    )
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--num-train-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", help="Train in bf16 autocast (if CUDA).")
    parser.add_argument(
        "--save-bf16",
        action="store_true",
        help="Also save a bf16 checkpoint as model.bf16.pt for faster CPU/GPU inference.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_chars < args.max_len:
        print(
            f"[warn] --max-chars ({args.max_chars}) < --max-len ({args.max_len});"
            " non-ASCII text may be truncated below the byte budget.",
            flush=True,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    autocast_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else None

    id2label, label2id = load_labels(args.data_root)
    num_classes = len(id2label)
    lang2idx = dict(label2id)

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
    if args.max_train_samples is not None:
        train_dataset.frame = train_dataset.frame.head(args.max_train_samples).copy()
    if args.max_eval_samples is not None:
        eval_dataset.frame = eval_dataset.frame.head(args.max_eval_samples).copy()

    if num_classes <= int(train_dataset.frame["language_label_id"].max()):
        raise ValueError(
            f"v1_languages.csv reports {num_classes} labels but train split has higher label ids."
        )

    collator = ByteCollator(max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    config = CONFIGS[args.config_name]
    model = ByteHybrid(num_classes=num_classes, max_len=args.max_len, **config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ByteHybrid({args.config_name}) | params={n_params/1e6:.2f}M | classes={num_classes} | max_len={args.max_len}", flush=True)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    metadata = {
        "model_arch": "ByteHybrid",
        "config_name": args.config_name,
        "config": config,
        "num_classes": num_classes,
        "max_len": args.max_len,
        "id2label": id2label,
        "label2id": label2id,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "snippet_config": asdict(snippet_config),
        "n_params": n_params,
        "device": str(device),
        "autocast_dtype": "bf16" if autocast_dtype is not None else "fp32",
    }
    with (args.output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (args.output_dir / "lang2idx.json").open("w", encoding="utf-8") as handle:
        json.dump(lang2idx, handle, indent=2, sort_keys=True)

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(args.num_train_epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_count = 0
        epoch_start = time.time()
        lr_now = args.learning_rate
        for step, batch in enumerate(train_loader):
            byte_ids = batch["byte_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            lr_now = cosine_warmup(
                global_step, total_steps, warmup_steps, args.learning_rate, args.min_learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(byte_ids)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(byte_ids)
                loss = loss_fn(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bsz = labels.size(0)
            running_loss += float(loss.item()) * bsz
            running_count += bsz
            global_step += 1

            if (step + 1) % args.logging_steps == 0:
                avg_loss = running_loss / max(running_count, 1)
                elapsed = time.time() - epoch_start
                print(
                    f"epoch {epoch} step {step + 1}/{steps_per_epoch} "
                    f"loss={avg_loss:.4f} lr={lr_now:.2e} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        train_loss = running_loss / max(running_count, 1)
        eval_metrics = evaluate(model, eval_loader, device, num_classes, autocast_dtype)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr_end": lr_now,
            "elapsed_seconds": time.time() - epoch_start,
            **eval_metrics,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={eval_metrics['accuracy']:.4f} val_macro_f1={eval_metrics['macro_f1']:.4f}",
            flush=True,
        )
        with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        improved = eval_metrics["macro_f1"] > best_f1 + args.early_stopping_threshold
        if improved:
            best_f1 = eval_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "model.pt",
                model,
                num_classes,
                args.max_len,
                lang2idx,
                args.config_name,
                dtype=None,
            )
            if args.save_bf16:
                save_checkpoint(
                    args.output_dir / "model.bf16.pt",
                    model,
                    num_classes,
                    args.max_len,
                    lang2idx,
                    args.config_name,
                    dtype=torch.bfloat16,
                )
            print(f"  saved best model (macro_f1={best_f1:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"early stopping: no improvement for {epochs_without_improvement} epochs"
                    f" (best macro_f1={best_f1:.4f} @ epoch {best_epoch})",
                    flush=True,
                )
                break

    print(f"done. best macro_f1={best_f1:.4f} @ epoch {best_epoch}", flush=True)
    summary = {
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] + 1 if history else 0,
        "history_path": "training_history.json",
    }
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
