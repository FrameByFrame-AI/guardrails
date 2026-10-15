#!/usr/bin/env python3
"""Fine-tune + evaluate a model on KLUE-NER (entity-level F1 on the dev set).

Same recipe/metric for any --model, so ko-ettin vs klue/roberta-base is apples-to-apples.
KLUE test labels are hidden, so we report on the validation split (standard practice).
"""
import argparse, os
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--max-length", type=int, default=256)
ap.add_argument("--out", default="/tmp/klue_ner_out")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
set_seed(args.seed)

ds = load_dataset("klue", "ner")
label_names = ds["train"].features["ner_tags"].feature.names
id2label = {i: n for i, n in enumerate(label_names)}
label2id = {n: i for i, n in enumerate(label_names)}
print(f"model={args.model} labels={label_names}")

tok = AutoTokenizer.from_pretrained(args.model)


def align(ex):
    text = "".join(ex["tokens"])
    enc = tok(text, truncation=True, max_length=args.max_length, return_offsets_mapping=True)
    cl = ex["ner_tags"]
    labels = []
    for s, e in enc["offset_mapping"]:
        labels.append(-100 if s == e else (cl[s] if s < len(cl) else -100))
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


tokenized = ds.map(align, remove_columns=ds["train"].column_names)

model = AutoModelForTokenClassification.from_pretrained(
    args.model, num_labels=len(label_names), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True)
if hasattr(model.config, "reference_compile"):
    model.config.reference_compile = False


def bio_spans(tags):
    spans, cur, st = set(), None, None
    for i, t in enumerate(tags):
        n = id2label[int(t)]
        if n == "O" or n.startswith("B-"):
            if cur is not None:
                spans.add((cur, st, i)); cur = None
        if n.startswith("B-"):
            cur, st = n[2:], i
        elif n.startswith("I-") and cur is None:
            cur, st = n[2:], i
    if cur is not None:
        spans.add((cur, st, len(tags)))
    return spans


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    tp = fp = fn = 0
    for pr, la in zip(preds, p.label_ids):
        keep = la != -100
        g = bio_spans(la[keep]); pp = bio_spans(pr[keep])
        tp += len(g & pp); fp += len(pp - g); fn += len(g - pp)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"entity_precision": round(P, 4), "entity_recall": round(R, 4), "entity_f1": round(F, 4)}


targs = TrainingArguments(
    output_dir=args.out, num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs, per_device_eval_batch_size=64,
    learning_rate=args.lr, lr_scheduler_type="linear", warmup_ratio=0.1, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", logging_steps=100, bf16=True,
    report_to=[], seed=args.seed, dataloader_num_workers=2)

trainer = Trainer(model=model, args=targs, train_dataset=tokenized["train"],
                  eval_dataset=tokenized["validation"],
                  data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics)
trainer.train()
m = trainer.evaluate()
print(f"RESULT model={args.model} entity_f1={m['eval_entity_f1']} "
      f"P={m['eval_entity_precision']} R={m['eval_entity_recall']}")
