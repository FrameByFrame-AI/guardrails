#!/usr/bin/env python3
"""LoRA contrastive finetune of the Granite multilingual encoder.

Bi-encoder InfoNCE: each batch contains N (sentence, topic) positive pairs;
all other topics in the batch act as negatives via
``MultipleNegativesRankingLoss``. The scanner uses cosine similarity at
inference time, so this directly trains the metric the scanner consumes.

Backbone: ``ibm-granite/granite-embedding-97m-multilingual-r2`` (ModernBERT,
12 layers, 384-dim). LoRA target modules are ``Wqkv``/``Wo`` (ModernBERT's
fused QKV and output projections — *not* BERT-style ``query/key/value``).

Per-epoch eval calls the BanTopicsScanner against the held-out 50-prompt
benchmark with the production default threshold (0.5).

    docker compose run --rm finetune-lora-bantopics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_pairs(paths: list[Path]) -> list[dict]:
    pairs: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "sentence" in row and "topic" in row:
                    pairs.append({"sentence": row["sentence"], "topic": row["topic"]})
    return pairs


def stratified_eval_split(pairs: list[dict], n_per_topic: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out N pairs per topic for held-in validation. Use a deterministic
    hash so the same row always lands on the same side across runs."""
    import hashlib

    by_topic: dict[str, list[dict]] = {}
    for p in pairs:
        by_topic.setdefault(p["topic"], []).append(p)

    train: list[dict] = []
    eval_: list[dict] = []
    for topic, rows in by_topic.items():
        ranked = sorted(
            rows,
            key=lambda r: hashlib.sha1(
                f"{seed}|{topic}|{r['sentence']}".encode("utf-8")
            ).hexdigest(),
        )
        eval_.extend(ranked[:n_per_topic])
        train.extend(ranked[n_per_topic:])
    return train, eval_


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", required=True, nargs="+", type=Path,
                   help="One or more JSONL files of {sentence, topic} pairs.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--base-model", default="ibm-granite/granite-embedding-97m-multilingual-r2")

    # LoRA — defaults match the privacy-filter recipe in this repo.
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+",
                   default=["Wqkv", "Wo"],
                   help="ModernBERT module names. Use ['query','key','value','dense'] for BERT.")

    # Optimisation.
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--bf16", action="store_true")

    # Eval.
    p.add_argument("--bantopics-eval-jsonl", type=Path, default=None,
                   help="Held-out 50-prompt benchmark; runs scanner-style eval per epoch.")
    p.add_argument("--bantopics-eval-threshold", type=float, default=0.5,
                   help="Production default threshold.")
    p.add_argument("--held-in-per-topic", type=int, default=20,
                   help="Pairs held back per topic for the in-distribution validation slice.")

    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--resume-from-checkpoint", type=Path, default=None)
    p.add_argument("--attn-implementation", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="Force the attention backend. flash_attention_2 silently "
                        "crashes on Blackwell (sm_120); eager is the safe default.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers. 0 starves the GPU on small batches.")
    args = p.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, inject_adapter_in_model
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
    )
    # New canonical import paths in sentence-transformers >=3.x.
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    pairs = load_pairs(args.pairs)
    print(f"# loaded {len(pairs)} (sentence, topic) pairs from {len(args.pairs)} file(s)")

    train_pairs, val_pairs = stratified_eval_split(pairs, args.held_in_per_topic, args.seed)
    print(f"# split: {len(train_pairs)} train / {len(val_pairs)} held-in val")

    # ModernBERT/Granite invokes torch.compile internally on the first forward.
    # PyTorch 2.11's dynamo + inductor doesn't have stable Blackwell (sm_120)
    # kernels — the JIT path SIGSEGVs (exit 139) on the first training step.
    # Mitigation is the ``TORCHDYNAMO_DISABLE=1`` env var set on the service
    # (see docker-compose.yml). The ``reference_compile=False`` model kwarg
    # would also work but isn't accepted by transformers 4.51's ModernBert.
    encoder = SentenceTransformer(
        args.base_model,
        model_kwargs={"attn_implementation": args.attn_implementation},
    )
    encoder.max_seq_length = args.max_seq_length

    transformer_module = encoder._first_module()
    base_model = transformer_module.auto_model

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    # Apply LoRA. PEFT's API varies across versions:
    #  - older PEFT returns a PeftModel wrapping the base
    #  - newer PEFT modifies the base in-place and returns it (still adds
    #    ``peft_config`` to the base)
    # We detect "did it work?" by trainable-param ratio, not isinstance —
    # an isinstance check produces false negatives on newer PEFT and triggers
    # a double-adapter on the fallback path.
    peft_wrapped = get_peft_model(base_model, lora_cfg)
    transformer_module.auto_model = peft_wrapped

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    if trainable == total:
        print("# get_peft_model did not freeze params — falling back to inject_adapter_in_model")
        inject_adapter_in_model(lora_cfg, transformer_module.auto_model, adapter_name="default")
        peft_wrapped = transformer_module.auto_model
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in encoder.parameters())

    if trainable == total:
        raise RuntimeError(
            f"LoRA wrapping failed: all {total:,} params are trainable. "
            f"Check that --target-modules ({args.target_modules}) match the "
            f"base model's linear layer names."
        )
    print(f"# LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    train_ds = Dataset.from_list(
        [{"anchor": r["sentence"], "positive": r["topic"]} for r in train_pairs]
    )
    val_ds = Dataset.from_list(
        [{"anchor": r["sentence"], "positive": r["topic"]} for r in val_pairs]
    )

    loss = MultipleNegativesRankingLoss(encoder)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        # Random sampler: with our 24-topic vocabulary, NO_DUPLICATES caps
        # batch size at 24 (no batch can have more than one row per topic).
        # The default sampler accepts duplicate positives — at batch 256 ~4%
        # of in-batch "negatives" share the anchor's topic, which MNRL treats
        # as a noisy gradient. The trade is well worth ~10× larger batches.
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=encoder,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
    )

    # Diagnostic: how many batches will the sampler actually deliver per epoch?
    try:
        n_batches = len(trainer.get_train_dataloader())
        print(f"# train dataloader batches/epoch = {n_batches}", flush=True)
    except Exception as exc:
        print(f"# could not size train dataloader: {exc!r}", flush=True)

    print("# >>> entering trainer.train()", flush=True)
    try:
        train_result = trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint else None
        )
        print(f"# <<< trainer.train() returned: {train_result}", flush=True)
    except Exception as exc:
        import traceback
        print(f"# !!! trainer.train() raised: {exc!r}", flush=True)
        traceback.print_exc()
        raise

    print("# >>> saving adapter + merged model", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "lora_adapter"
    if hasattr(peft_wrapped, "save_pretrained"):
        peft_wrapped.save_pretrained(str(adapter_dir))
    else:
        from peft.utils.save_and_load import get_peft_model_state_dict
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            get_peft_model_state_dict(peft_wrapped, adapter_name="default"),
            adapter_dir / "adapter_model.bin",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(lora_cfg.to_dict(), indent=2)
        )
    encoder.save(str(args.output_dir / "merged_for_inference"),
                 safe_serialization=True)
    print(f"# adapter → {adapter_dir}", flush=True)
    print(f"# merged-for-inference → {args.output_dir / 'merged_for_inference'}", flush=True)

    # Sectioned eval — production_50 is the headline; xstest_safe is the
    # over-refusal regression; harmbench/jbb/dna held-outs are in-distribution
    # validation; safetybench_zh is multilingual transfer signal.
    if args.bantopics_eval_jsonl and args.bantopics_eval_jsonl.exists():
        from collections import defaultdict
        from finetune_bantopics.scanner import (
            BanTopicsScanner,
            ScanRequest,
            ScannerConfig,
        )

        cfg = ScannerConfig(
            model_name_or_path=str(args.output_dir / "merged_for_inference"),
        )
        scanner = BanTopicsScanner(cfg)

        rows = []
        with open(args.bantopics_eval_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        sections: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            section = r.get("eval_section", "unspecified")
            req = ScanRequest(
                scanner_name="BanTopics",
                scanner_type="prompt",
                scanner_params={
                    "threshold": args.bantopics_eval_threshold,
                    "topics": r["topics"],
                },
                prompt=r["prompt"],
            )
            resp = scanner.scan(req)
            blocked = not resp.is_valid
            expected_block = r["expected"] == "BLOCK"
            sections[section].append({
                "expected_block": expected_block,
                "actual_block": blocked,
                "score": resp.score,
            })

        per_section = {}
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
                # For PASS-only sections (xstest_safe), pass_rate is the metric
                # that matters; tn / (tn + fp) is "correctly let through".
                "pass_rate": tn / max(tn + fp, 1) if (tn + fp) else None,
            }

        result = {
            "threshold": args.bantopics_eval_threshold,
            "sections": per_section,
        }
        (args.output_dir / "benchmark_eval.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
