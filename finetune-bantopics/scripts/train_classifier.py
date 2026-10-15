#!/usr/bin/env python3
"""Multi-label classifier head over Granite-97m, LoRA-tuned on the 24 production topics.

Replaces the bi-encoder + InfoNCE path. With a fixed label set, supervised
multi-label sigmoid + BCE is structurally cleaner than contrastive cosine:
- No NO_DUPLICATES sampler ceiling — batch size only limited by VRAM
- Each topic has a dedicated head row, no representation budget competition
- Sigmoid outputs are real probabilities, threshold semantics align with the API
- One forward per request at inference (no topic encoding, no topic cache)

    docker compose run --rm train-classifier-bantopics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_multilabel(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "sentence" in row and "labels" in row:
                    rows.append({"sentence": row["sentence"], "labels": row["labels"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, nargs="+", type=Path,
                   help="One or more multi-label JSONLs (from collapse_pairs_to_multilabel.py).")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--base-model", default="ibm-granite/granite-embedding-97m-multilingual-r2")

    # LoRA — same target modules as the bi-encoder (ModernBERT Wqkv/Wo).
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+", default=["Wqkv", "Wo"])

    # Optimisation — classifier-friendly defaults; batch can be much bigger now.
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)

    # Eval.
    p.add_argument("--eval-suite", type=Path, default=None)
    p.add_argument("--eval-threshold", type=float, default=0.5,
                   help="Sigmoid threshold for the post-training eval. Production default is 0.5.")
    p.add_argument("--val-fraction", type=float, default=0.01,
                   help="Fraction of training data to hold out as held-in val.")

    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--attn-implementation", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--early-stopping-patience", type=int, default=3,
                   help="Stop if eval_loss hasn't improved in N consecutive epochs.")
    p.add_argument("--early-stopping-threshold", type=float, default=0.0,
                   help="Minimum improvement to count as progress (0 = any improvement).")
    args = p.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, inject_adapter_in_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from finetune_bantopics.topics import all_topics

    topic_list = all_topics()
    n_labels = len(topic_list)
    print(f"# label space: {n_labels} topics")

    # Data.
    rows = load_multilabel(args.data)
    print(f"# loaded {len(rows)} (sentence, labels) rows")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    n_val = max(64, int(len(rows) * args.val_fraction))
    val_rows = [rows[i] for i in perm[:n_val]]
    train_rows = [rows[i] for i in perm[n_val:]]
    print(f"# split: {len(train_rows)} train / {len(val_rows)} val")

    # Tokenizer + model.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=n_labels,
        problem_type="multi_label_classification",
        id2label={i: t for i, t in enumerate(topic_list)},
        label2id={t: i for i, t in enumerate(topic_list)},
        attn_implementation=args.attn_implementation,
    )

    # LoRA on the encoder; classifier head stays fully trainable (it's tiny).
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["classifier", "score"],
    )
    peft_wrapped = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in peft_wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_wrapped.parameters())
    if trainable == total:
        print("# get_peft_model didn't freeze — falling back to inject_adapter_in_model")
        inject_adapter_in_model(lora_cfg, model, adapter_name="default")
        peft_wrapped = model
        trainable = sum(p.numel() for p in peft_wrapped.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_wrapped.parameters())
    if trainable == total:
        raise RuntimeError(f"LoRA wrapping failed: all {total:,} params trainable.")
    print(f"# trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Tokenize + assemble HF Datasets.
    def to_dataset(rows: list[dict]) -> Dataset:
        ds = Dataset.from_list(rows)
        def tok(batch):
            enc = tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )
            enc["labels"] = [list(map(float, lbl)) for lbl in batch["labels"]]
            return enc
        return ds.map(tok, batched=True, remove_columns=["sentence"])

    train_ds = to_dataset(train_rows)
    val_ds = to_dataset(val_rows)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # Early stopping — monitors eval_loss on the held-in val slice. Set
        # max num_train_epochs high (e.g. 100) and let the patience cut off
        # when the loss plateaus.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,  # labels are already in the encoded dict
    )

    trainer = Trainer(
        model=peft_wrapped,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )],
    )

    print(f"# train batches/epoch ≈ {len(train_ds) // args.batch_size}", flush=True)
    print("# >>> entering trainer.train()", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"# <<< trainer.train() returned ({time.time() - t0:.1f}s)", flush=True)

    # Save adapter + merged model.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "lora_adapter"
    if hasattr(peft_wrapped, "save_pretrained"):
        peft_wrapped.save_pretrained(str(adapter_dir))

    merged_dir = args.output_dir / "merged_for_inference"
    if isinstance(peft_wrapped, PeftModel):
        merged = peft_wrapped.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
    else:
        peft_wrapped.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"# adapter → {adapter_dir}", flush=True)
    print(f"# merged → {merged_dir}", flush=True)

    # Sectioned eval against the eval suite.
    if args.eval_suite and args.eval_suite.exists():
        print("# >>> running per-section eval", flush=True)
        from transformers import AutoModelForSequenceClassification as AMSC

        eval_model = AMSC.from_pretrained(
            str(merged_dir),
            attn_implementation=args.attn_implementation,
        ).to(model.device)
        eval_model.eval()
        eval_tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))

        from finetune_bantopics.splitter import split_sentences
        from finetune_bantopics.topics import all_topics as _all_topics
        topic_idx = {t: i for i, t in enumerate(_all_topics())}

        suite_rows = []
        with open(args.eval_suite, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    suite_rows.append(json.loads(line))

        sections: dict[str, list[dict]] = defaultdict(list)
        with torch.no_grad():
            for r in suite_rows:
                sentences = split_sentences(
                    r["prompt"],
                    language=r.get("language", "en"),
                    backend="pysbd",
                ) or [r["prompt"]]
                enc = eval_tokenizer(
                    sentences, padding=True, truncation=True,
                    max_length=args.max_seq_length, return_tensors="pt",
                ).to(eval_model.device)
                logits = eval_model(**enc).logits     # S × n_labels
                probs = torch.sigmoid(logits)         # S × n_labels

                # Filter to requested topics; if a topic isn't in the catalog,
                # it contributes 0 (never blocks). String normalised lightly.
                idxs = []
                missing = []
                for t in r["topics"]:
                    norm = t.strip().lower().replace("-", " ").replace("_", " ")
                    if norm in topic_idx:
                        idxs.append(topic_idx[norm])
                    else:
                        missing.append(t)
                if not idxs:
                    score = 0.0
                else:
                    score = float(probs[:, idxs].max().item())

                blocked = score >= args.eval_threshold
                expected_block = r["expected"] == "BLOCK"
                sections[r.get("eval_section", "?")].append({
                    "expected_block": expected_block,
                    "actual_block": blocked,
                    "score": score,
                    "missing_topics": missing,
                })

        per_section = {}
        all_scores: list[float] = []
        for section, entries in sections.items():
            tp = sum(1 for e in entries if e["expected_block"] and e["actual_block"])
            fn = sum(1 for e in entries if e["expected_block"] and not e["actual_block"])
            fp = sum(1 for e in entries if not e["expected_block"] and e["actual_block"])
            tn = sum(1 for e in entries if not e["expected_block"] and not e["actual_block"])
            n = tp + fp + tn + fn
            per_section[section] = {
                "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "accuracy": (tp + tn) / max(n, 1),
                "pass_rate": tn / max(tn + fp, 1) if (tn + fp) else None,
            }
            all_scores.extend(e["score"] for e in entries)

        result = {
            "threshold": args.eval_threshold,
            "score_distribution": {
                "min": min(all_scores) if all_scores else 0,
                "p50": sorted(all_scores)[len(all_scores) // 2] if all_scores else 0,
                "max": max(all_scores) if all_scores else 0,
            },
            "sections": per_section,
        }
        (args.output_dir / "benchmark_eval.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
