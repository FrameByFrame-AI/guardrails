#!/usr/bin/env python3
"""Train ByteHybrid as a 25-language multi-label code-language detector.

Trained on Accuknoxtechnologies/CodeLanguage (multi-label parquet with prompts
that mix natural language and code). Output shape matches the dataset's
`target` field exactly:

    {"is_valid": bool, "category": {"<Lang>": true, ...}}

`is_valid` is derived at inference time from a threshold rule (max per-language
sigmoid prob < threshold => is_valid=false, category={}). Training uses
BCEWithLogitsLoss; benign rows push all 25 logits down.

Warm-starts from the 107-class lite checkpoint by default — encoder weights
are fully reused, and 19/25 head rows are initialized from the matching old
softmax weights (the new shell/config languages get fresh init).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_language_id.byte_hybrid import CONFIGS, ByteHybrid
from code_language_id.multilabel_dataset import (
    LANGUAGES,
    NUM_LANGUAGES,
    CodeLanguageMultilabelDataset,
)


DEFAULT_DATA_DIR = Path(
    "~/llm_models/Accuknoxtechnologies-CodeLanguage/data"
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/llm_models/guardrail_code_models/programming-language-identification-25-multilabel"
).expanduser()
DEFAULT_WARM_START = Path(
    "~/llm_models/guardrail_code_models/"
    "programming-language-identification-100plus-lite/model.pt"
).expanduser()

# Old-checkpoint vocab names that map to the new label set with case/spelling
# normalization. Anything not listed and not an exact string match gets fresh
# init in the head.
OLD_NAME_ALIASES: dict[str, str] = {
    "AWK": "Awk",
    "Batch": "Batchfile",
}


def encode_bytes(texts: list[str], max_len: int) -> torch.Tensor:
    """UTF-8 byte encoding; pad with byte 256, identical to CommonLingua."""
    out = np.full((len(texts), max_len), 256, dtype=np.int64)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        raw = t.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


class ByteMultilabelCollator:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        byte_ids = encode_bytes([item["text"] for item in batch], self.max_len)
        labels = torch.from_numpy(np.stack([item["labels"] for item in batch], axis=0))
        is_valid = torch.tensor([item["is_valid"] for item in batch], dtype=torch.bool)
        return {"byte_ids": byte_ids, "labels": labels, "is_valid": is_valid}


class FocalBCEWithLogitsLoss(torch.nn.Module):
    """Focal loss applied to each sigmoid logit independently.

    For each (batch, class) element:
        p_t  = sigmoid(logit) if target==1 else 1 - sigmoid(logit)
        a_t  = alpha          if target==1 else 1 - alpha
        loss = -a_t * (1 - p_t) ** gamma * log(p_t)

    With alpha=0.5 and gamma=0 this collapses to mean BCE. We default to
    alpha=0.25, gamma=2.0 (the values from the original focal-loss paper —
    they bias the loss toward hard examples and toward the rarer positive
    class, which directly addresses the "missing the secondary language"
    failure mode on multi-label code prompts).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use the numerically-stable bce_with_logits as the base term, then scale.
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return focal.mean()


def build_loss_fn(
    loss_type: str,
    train_labels: np.ndarray,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Returns (loss_module, info_dict) where info_dict records hyperparameters."""
    info: dict[str, Any] = {"loss_type": loss_type}
    if loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss(), info
    if loss_type == "weighted_bce":
        # per-class pos_weight = num_negatives / num_positives
        pos = train_labels.sum(axis=0)
        neg = train_labels.shape[0] - pos
        pos_weight = neg / np.maximum(pos, 1)
        info["pos_weight_per_class"] = pos_weight.tolist()
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pw), info
    if loss_type == "focal":
        info["focal_alpha"] = focal_alpha
        info["focal_gamma"] = focal_gamma
        return FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma), info
    raise ValueError(f"unknown loss_type {loss_type!r}")


def cosine_warmup(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if total_steps <= warmup_steps:
        return peak_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def warm_start_from_checkpoint(model: ByteHybrid, checkpoint_path: Path) -> dict[str, Any]:
    """Copy encoder weights and matching head rows from a ByteHybrid checkpoint.

    Tolerates source checkpoints that lack ``attn_pool_proj`` weights (the new
    attention-pool layer) — those stay at their zero init, which is identical
    to mean pool at construction.

    Returns a small report describing what was loaded.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_state: dict[str, torch.Tensor] = ckpt["model_state_dict"]
    old_lang2idx: dict[str, int] = ckpt["lang2idx"]
    old_num_classes: int = ckpt["num_classes"]

    # Final classifier row depends on num_classes; handle separately.
    HEAD_FINAL_W = "head.3.weight"
    HEAD_FINAL_B = "head.3.bias"
    encoder_state = {k: v for k, v in old_state.items() if k not in (HEAD_FINAL_W, HEAD_FINAL_B)}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    # Filter out keys we expect to be missing/different:
    #  - head final layer (num_classes-dependent, copied row-by-row below)
    #  - attn_pool_proj.{weight,bias}: new layer, absent in pre-attention-pool ckpts
    expected_missing = {HEAD_FINAL_W, HEAD_FINAL_B, "attn_pool_proj.weight", "attn_pool_proj.bias"}
    missing_real = [k for k in missing if k not in expected_missing]
    if missing_real or unexpected:
        raise RuntimeError(
            f"warm-start mismatch — missing={missing_real} unexpected={unexpected}"
        )

    # Warm-start matching rows of the final classifier.
    with torch.no_grad():
        old_w: torch.Tensor = old_state[HEAD_FINAL_W]  # [old_num_classes, d]
        old_b: torch.Tensor = old_state[HEAD_FINAL_B]  # [old_num_classes]
        new_w: torch.Tensor = model.head[-1].weight     # [25, d]
        new_b: torch.Tensor = model.head[-1].bias       # [25]
        if old_w.shape[0] != old_num_classes:
            raise RuntimeError(
                f"old checkpoint head shape {old_w.shape} disagrees with num_classes={old_num_classes}"
            )

        matched: list[str] = []
        unmatched: list[str] = []
        for new_idx, name in enumerate(LANGUAGES):
            old_name = name if name in old_lang2idx else OLD_NAME_ALIASES.get(name)
            if old_name is None or old_name not in old_lang2idx:
                unmatched.append(name)
                continue
            old_idx = old_lang2idx[old_name]
            new_w[new_idx].copy_(old_w[old_idx])
            new_b[new_idx].copy_(old_b[old_idx])
            matched.append(name)

    return {
        "checkpoint": str(checkpoint_path),
        "old_num_classes": old_num_classes,
        "old_max_len": ckpt.get("max_len"),
        "encoder_loaded": True,
        "head_rows_matched": matched,
        "head_rows_unmatched": unmatched,
    }


@dataclass
class EvalReport:
    bce_loss: float
    is_valid_threshold: float
    is_valid_accuracy: float
    macro_f1: float
    micro_f1: float
    per_class_f1: dict[str, float]
    json_exact_match: float
    num_eval: int


@torch.no_grad()
def evaluate(
    model: ByteHybrid,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    autocast_dtype: torch.dtype | None,
) -> EvalReport:
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_is_valid: list[np.ndarray] = []
    loss_sum = 0.0
    n_elements = 0

    for batch in loader:
        byte_ids = batch["byte_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(byte_ids)
        else:
            logits = model(byte_ids)
        logits = logits.float()
        loss_sum += float(bce(logits, labels).item())
        n_elements += labels.numel()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
        all_is_valid.append(batch["is_valid"].numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(bool)
    is_valid_true = np.concatenate(all_is_valid, axis=0)

    preds = probs >= threshold  # [N, 25]
    pred_is_valid = preds.any(axis=1)
    is_valid_acc = float((pred_is_valid == is_valid_true).mean())

    # Per-class precision/recall/F1 — skip classes with no positives in eval.
    per_class: dict[str, float] = {}
    f1s: list[float] = []
    tp_total = fp_total = fn_total = 0
    for c, lang in enumerate(LANGUAGES):
        tp = int(np.logical_and(preds[:, c], labels[:, c]).sum())
        fp = int(np.logical_and(preds[:, c], ~labels[:, c]).sum())
        fn = int(np.logical_and(~preds[:, c], labels[:, c]).sum())
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[lang] = f1
        f1s.append(f1)
    micro_prec = tp_total / max(tp_total + fp_total, 1)
    micro_recall = tp_total / max(tp_total + fn_total, 1)
    micro_f1 = (
        0.0
        if micro_prec + micro_recall == 0
        else 2 * micro_prec * micro_recall / (micro_prec + micro_recall)
    )

    # Strict per-row JSON match: pred_is_valid == is_valid AND label set matches.
    exact_rows = 0
    for i in range(len(preds)):
        if pred_is_valid[i] != is_valid_true[i]:
            continue
        if not is_valid_true[i] and not pred_is_valid[i]:
            exact_rows += 1
            continue
        if np.array_equal(preds[i], labels[i]):
            exact_rows += 1
    exact_match = exact_rows / max(len(preds), 1)

    return EvalReport(
        bce_loss=loss_sum / max(n_elements, 1),
        is_valid_threshold=threshold,
        is_valid_accuracy=is_valid_acc,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        micro_f1=float(micro_f1),
        per_class_f1=per_class,
        json_exact_match=float(exact_match),
        num_eval=int(len(preds)),
    )


def save_checkpoint(
    path: Path,
    model: ByteHybrid,
    max_len: int,
    config_name: str,
    threshold: float,
    pool_type: str = "mean",
    dtype: torch.dtype | None = None,
) -> None:
    state = model.state_dict()
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    payload = {
        "model_state_dict": state,
        "num_classes": NUM_LANGUAGES,
        "max_len": max_len,
        "lang2idx": {lang: i for i, lang in enumerate(LANGUAGES)},
        "config": config_name,
        "pool_type": pool_type,
        "is_valid_threshold": threshold,
        "task": "multilabel",
    }
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing train-*.parquet and test-*.parquet.")
    parser.add_argument("--train-parquet", type=Path, default=None,
                        help="Override path to train parquet (default: data-dir/train-00000-of-00001.parquet).")
    parser.add_argument("--eval-parquet", type=Path, default=None,
                        help="Override path to eval parquet (default: data-dir/test-00000-of-00001.parquet).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warm-start-from", type=Path, default=DEFAULT_WARM_START,
                        help="Old ByteHybrid checkpoint to copy encoder weights from. "
                             "Pass empty string to disable.")
    parser.add_argument("--config-name", default="base_ngram", choices=sorted(CONFIGS.keys()))
    parser.add_argument("--pool-type", default="mean", choices=["mean", "attention"],
                        help="Sequence pooling. 'attention' adds a learned softmax pool.")
    parser.add_argument("--loss-type", default="bce", choices=["bce", "weighted_bce", "focal"])
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-len", type=int, default=2048)

    def _str2bool(v: str) -> bool:
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    parser.add_argument("--variable-window", type=_str2bool, default=False,
                        help="Resample single/benign rows to a random byte window each epoch "
                             "(multi rows are left alone to preserve label correctness).")
    parser.add_argument("--variable-window-min-chars", type=int, default=256)
    parser.add_argument("--variable-window-max-chars", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for per-language predictions and is_valid rule.")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Lower than from-scratch default (3e-3) because we warm-start.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-bf16", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = (args.train_parquet or args.data_dir / "train-00000-of-00001.parquet").expanduser()
    eval_path = (args.eval_parquet or args.data_dir / "test-00000-of-00001.parquet").expanduser()
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    autocast_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else None

    train_dataset = CodeLanguageMultilabelDataset(
        train_path,
        variable_window=args.variable_window,
        min_chars=args.variable_window_min_chars,
        max_chars=args.variable_window_max_chars,
        seed=args.seed,
    )
    eval_dataset = CodeLanguageMultilabelDataset(eval_path)
    if args.max_train_samples is not None:
        train_dataset.frame = train_dataset.frame.head(args.max_train_samples).copy()
        train_dataset._labels = train_dataset._labels[: args.max_train_samples]
        train_dataset._is_valid = train_dataset._is_valid[: args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_dataset.frame = eval_dataset.frame.head(args.max_eval_samples).copy()
        eval_dataset._labels = eval_dataset._labels[: args.max_eval_samples]
        eval_dataset._is_valid = eval_dataset._is_valid[: args.max_eval_samples]

    collator = ByteMultilabelCollator(max_len=args.max_len)
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
    model = ByteHybrid(
        num_classes=NUM_LANGUAGES,
        max_len=args.max_len,
        pool_type=args.pool_type,
        **config,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"ByteHybrid({args.config_name}) | params={n_params/1e6:.2f}M | "
        f"classes={NUM_LANGUAGES} | max_len={args.max_len} | pool={args.pool_type}",
        flush=True,
    )

    warm_report: dict[str, Any] | None = None
    if args.warm_start_from and str(args.warm_start_from) != "":
        warm_path = Path(args.warm_start_from).expanduser()
        if not warm_path.exists():
            print(f"[warn] warm-start checkpoint not found: {warm_path} — training from scratch.", flush=True)
        else:
            warm_report = warm_start_from_checkpoint(model, warm_path)
            model = model.to(device)
            print(
                f"warm-start: encoder from {warm_path}, "
                f"head matched {len(warm_report['head_rows_matched'])}/{NUM_LANGUAGES} "
                f"(unmatched: {warm_report['head_rows_unmatched']})",
                flush=True,
            )

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
    loss_fn, loss_info = build_loss_fn(
        args.loss_type,
        train_dataset._labels,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        device=device,
    )
    print(f"loss: {loss_info}", flush=True)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    metadata: dict[str, Any] = {
        "model_arch": "ByteHybrid",
        "task": "multilabel",
        "config_name": args.config_name,
        "config": config,
        "num_classes": NUM_LANGUAGES,
        "languages": list(LANGUAGES),
        "max_len": args.max_len,
        "threshold": args.threshold,
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
        "n_params": n_params,
        "device": str(device),
        "autocast_dtype": "bf16" if autocast_dtype is not None else "fp32",
        "warm_start": warm_report,
        "train_parquet": str(train_path),
        "eval_parquet": str(eval_path),
        "pool_type": args.pool_type,
        "loss": loss_info,
    }
    with (args.output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (args.output_dir / "lang2idx.json").open("w", encoding="utf-8") as handle:
        json.dump({lang: i for i, lang in enumerate(LANGUAGES)}, handle, indent=2)

    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
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
        report = evaluate(model, eval_loader, device, args.threshold, autocast_dtype)
        # Pick early-stopping signal: prefer exact-row JSON match (the user-visible
        # success criterion). Fall back to macro F1 if tied.
        score = report.json_exact_match + 0.001 * report.macro_f1
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr_end": lr_now,
            "elapsed_seconds": time.time() - epoch_start,
            "eval_bce_loss": report.bce_loss,
            "eval_is_valid_accuracy": report.is_valid_accuracy,
            "eval_macro_f1": report.macro_f1,
            "eval_micro_f1": report.micro_f1,
            "eval_json_exact_match": report.json_exact_match,
            "eval_per_class_f1": report.per_class_f1,
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_is_valid_acc={report.is_valid_accuracy:.4f} "
            f"val_macro_f1={report.macro_f1:.4f} "
            f"val_micro_f1={report.micro_f1:.4f} "
            f"val_json_exact={report.json_exact_match:.4f}",
            flush=True,
        )
        with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        improved = score > best_score + args.early_stopping_threshold
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "model.pt",
                model,
                args.max_len,
                args.config_name,
                args.threshold,
                pool_type=args.pool_type,
                dtype=None,
            )
            if args.save_bf16:
                save_checkpoint(
                    args.output_dir / "model.bf16.pt",
                    model,
                    args.max_len,
                    args.config_name,
                    args.threshold,
                    pool_type=args.pool_type,
                    dtype=torch.bfloat16,
                )
            print(f"  saved best model (json_exact={report.json_exact_match:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"early stopping: no improvement for {epochs_without_improvement} epochs"
                    f" (best score={best_score:.4f} @ epoch {best_epoch})",
                    flush=True,
                )
                break

    summary = {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] + 1 if history else 0,
        "history_path": "training_history.json",
    }
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"done. best score={best_score:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
