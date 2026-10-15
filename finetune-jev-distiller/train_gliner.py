"""Fine-tune GLiNER2.5-Decide on verified gliner records (block + risk tasks) from to_gliner.py.

    python train_gliner.py --train data/split_train.jsonl --val data/split_val.jsonl --out runs/gold-gliner

Checkpoints go to <out>/checkpoints/{best,final}; the best is picked by loss on --val.
Training prompts are made identical to inference prompts (see configure_processor), and the run aborts
if any micro-batch fails instead of silently skipping it.
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger("train")


def read_jsonl(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def configure_processor(proc):
    """Match training prompts to inference, where all 24 labels are sent with real names and descriptions.

    gliner2's defaults would drop up to half the labels (sometimes the true one), rename labels to
    "label 1..N" half the time, and drop the descriptions in some prompts.
    Label order is still shuffled so the model cannot learn a position.
    """
    from gliner2.processor import SamplingConfig

    proc.sampling_config = SamplingConfig(
        remove_classification_prob=0.0,
        shuffle_classification_labels=True,
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
    )
    proc.__class__ = _describing_class(type(proc))
    return proc


_DESCRIBING: dict = {}


def _describing_class(base):
    """Module-level subclass (picklable for DataLoader workers) that always includes label descriptions."""
    if base.__name__.startswith("Describing"):
        return base
    if base not in _DESCRIBING:
        def _transform_schema(self, parent, fields, child_prefix, prompt=None, examples=None,
                              label_descriptions=None, example_mode="both"):
            if child_prefix == self.L_TOKEN and label_descriptions:
                example_mode = "both"
            return base._transform_schema(self, parent, fields, child_prefix, prompt=prompt, examples=examples,
                                          label_descriptions=label_descriptions, example_mode=example_mode)

        cls = type(f"Describing{base.__name__}", (base,), {"_transform_schema": _transform_schema, "__module__": __name__})
        _DESCRIBING[base] = cls
        globals()[cls.__name__] = cls
    return _DESCRIBING[base]


def check_prompt_parity(proc, record: dict, trials: int = 20) -> None:
    """Fail fast unless training builds the same schema prompt as inference (up to label order)."""
    captured = []
    real = proc._transform_schema

    def spy(*a, **kw):
        out = real(*a, **kw)
        captured.append(out)
        return out

    def canonical(schemas):
        return sorted((s[2].split(f" {proc.DESC_TOKEN} ")[0],
                       sorted(s[2].split(f" {proc.DESC_TOKEN} ")[1:]),
                       sorted(s[i + 1] for i, t in enumerate(s) if t == proc.L_TOKEN)) for s in schemas)

    proc._transform_schema = spy
    was = proc.is_training
    try:
        proc.is_training = False
        captured.clear()
        proc._process_classifications(json.loads(json.dumps(record["output"])), [], [], [], None)
        want = canonical(captured)
        proc.is_training = True
        for _ in range(trials):
            captured.clear()
            proc._process_classifications(json.loads(json.dumps(record["output"])), [], [], [], proc.sampling_config)
            if canonical(captured) != want:
                raise RuntimeError(f"training prompt differs from inference prompt:\n train {canonical(captured)}\n infer {want}")
    finally:
        proc.is_training = was
        del proc._transform_schema
    log.info(f"prompt parity OK over {trials} sampled training prompts")


def guard_failed_batches(trainer, limit: int) -> dict:
    state = {"failed": 0}
    original = trainer._record_failed_batch

    def record(*a, **kw):
        original(*a, **kw)
        state["failed"] += 1
        if state["failed"] > limit:
            raise RuntimeError(f"{state['failed']} micro-batches failed (limit {limit}); last error: {kw.get('error')!r}")

    trainer._record_failed_batch = record
    return state


def keep_last_batch(trainer) -> None:
    """gliner2 sets drop_last=True whenever the training set exceeds one batch, silently skipping up to
    batch_size-1 examples per epoch. Rebuild its training loader with drop_last=False."""
    from torch.utils.data import DataLoader

    original = trainer._create_dataloader

    def create(dataset, batch_size, shuffle=True, is_training=True):
        loader = original(dataset, batch_size, shuffle=shuffle, is_training=is_training)
        if not loader.drop_last:
            return loader
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=loader.sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            prefetch_factor=loader.prefetch_factor if loader.num_workers > 0 else None,
            collate_fn=loader.collate_fn,
            drop_last=False,
            persistent_workers=loader.persistent_workers,
        )

    trainer._create_dataloader = create


def preflight(args, train: list[dict]) -> None:
    """One optimizer step on the longest training batch (by subword tokens), on a throwaway model, so a batch
    that does not fit in memory fails in the first minute rather than partway through the run. Length is
    measured in tokens, not words: a 19-word run of punctuation can be 10k tokens."""
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    model = AutoExtractor.from_pretrained(args.base)
    tok = model.processor.tokenizer
    longest = sorted(train, key=lambda r: -len(tok(r["input"].lower(), add_special_tokens=False)["input_ids"]))
    longest = longest[: args.batch_size * args.grad_accum]
    configure_processor(model.processor)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = GLiNER2Trainer(model, TrainingConfig(
        output_dir="/tmp/jev-preflight", max_steps=1, batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, eval_strategy="no", save_best=False, num_workers=0,
        bf16=args.bf16, fp16=False, max_len=args.max_len))
    keep_last_batch(trainer)
    guard_failed_batches(trainer, 0)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train(train_data=longest)
    except RuntimeError as e:
        raise SystemExit(f"preflight failed on the longest batch (max_len={args.max_len}): {e}\n"
                         f"lower --max-len or --batch-size, or add --gradient-checkpointing") from None
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    log.info(f"preflight OK: longest batch (max_len={args.max_len}) peaked at {peak:.1f} GiB")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="/models/GLiNER2.5-Decide")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=5e-4)
    ap.add_argument("--eval-steps", type=int, default=0,
                    help="0 = validate each epoch (default); N = every N optimizer steps")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="truncate prompts to this many gliner2 word units (0 = no truncation)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-failed-batches", type=int, default=0)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gradient-checkpointing", action="store_true", help="trade speed for memory on small GPUs")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--keep-all-checkpoints", action="store_true",
                    help="keep per-epoch and final checkpoints (~1.9 GB each); default keeps only the one eval uses")
    args = ap.parse_args()
    args.max_len = args.max_len or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from gliner2 import AutoExtractor
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    train, val = read_jsonl(args.train), read_jsonl(args.val)
    if not train or not val:
        raise SystemExit(f"need non-empty train and val (got {len(train)} / {len(val)})")
    out = Path(args.out)
    if (out / "train-summary.json").exists():
        log.info(f"{out} is already trained; nothing to do (delete it to retrain)")
        return
    # gliner2's trainer cannot resume mid-run: a crash restarts this run from step 0 (per-epoch checkpoints only).
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"train {len(train)} val {len(val)} base {args.base}")
    if not args.skip_preflight:
        preflight(args, train)

    model = AutoExtractor.from_pretrained(args.base)
    check_prompt_parity(configure_processor(model.processor), train[0])
    if args.gradient_checkpointing:
        # gliner2's own flag uses the reentrant default, which fails on DeBERTa's shared relative-position embeddings
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = TrainingConfig(
        output_dir=str(out / "checkpoints"),
        experiment_name="jev-gliner",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        warmup_ratio=0.1,
        scheduler_type="linear",
        # gliner2 applies an epoch's trailing partial accumulation outside its step-eval check, so step-based
        # eval can skip steps; epoch eval always runs after the last update of each epoch.
        eval_strategy="steps" if args.eval_steps > 0 else "epoch",
        eval_steps=max(args.eval_steps, 1),
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        num_workers=args.workers,
        seed=args.seed,
        bf16=args.bf16,
        fp16=False,
        max_len=args.max_len,
    )
    trainer = GLiNER2Trainer(model, config)
    keep_last_batch(trainer)
    failed = guard_failed_batches(trainer, args.max_failed_batches)
    t0 = time.time()
    summary = trainer.train(train_data=train, eval_data=val)
    ckpts = out / "checkpoints"
    best = ckpts / "best" if (ckpts / "best").exists() else ckpts / "final"
    if not args.keep_all_checkpoints:
        for d in ckpts.iterdir():
            if d.is_dir() and d != best and (d.name == "final" or d.name.startswith("checkpoint-")):
                shutil.rmtree(d)
    info = {
        "base": args.base,
        "train_examples": len(train),
        "val_examples": len(val),
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "encoder_lr": args.encoder_lr,
        "task_lr": args.task_lr,
        "max_len": args.max_len,  # gliner_caller.py applies the same truncation
        "minutes": round((time.time() - t0) / 60, 2),
        "failed_micro_batches": failed["failed"],
        "best_checkpoint": str(best),
        "trainer_summary": {k: v for k, v in (summary or {}).items() if isinstance(v, (int, float, str))},
    }
    (out / "train-summary.json").write_text(json.dumps(info, indent=2))
    log.info(f"done: best checkpoint {best}")


if __name__ == "__main__":
    main()
