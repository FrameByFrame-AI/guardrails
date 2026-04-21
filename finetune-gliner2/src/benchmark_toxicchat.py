#!/usr/bin/env python3
"""
Benchmark the guardrail-gliner2-korean-english model on ToxicChat (lmsys/toxic-chat).

ToxicChat is English real user-LLM conversations with two binary labels:
  - toxicity: 0/1
  - jailbreaking: 0/1

Mapping to our schema:
  - toxicity=1        -> safety="unsafe"  AND  harmful has any non-"none" label
  - jailbreaking=1    -> adversarial has any non-"none" label

We evaluate:
  - safety accuracy (single-label)
  - adversarial binary F1 (any-label vs none)
  - harmful binary F1 (any-label vs none)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

from gliner2 import GLiNER2
from huggingface_hub import hf_hub_download


SAFETY_LABELS = ["safe", "unsafe"]

ADVERSARIAL_LABELS = [
    "jailbreak", "prompt_injection", "indirect_injection",
    "instruction_override", "data_exfiltration", "none",
]

HARMFUL_LABELS = [
    "violence", "criminal_planning", "hate_speech", "harassment",
    "sexual_content", "child_exploitation", "self_harm", "weapons",
    "drugs", "profanity", "fraud", "misinformation", "malware",
    "unauthorized_advice", "copyright", "none",
]


def binary_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/guardrail-gliner2-v4")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", default="/data/benchmark_results/toxicchat_guardrail_gliner2.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=4000,
                        help="truncate user_input to this many chars")
    args = parser.parse_args()

    print(f"Loading ToxicChat split={args.split}...")
    csv_path = hf_hub_download(
        repo_id="lmsys/toxic-chat",
        filename=f"data/0124/toxic-chat_annotation_{args.split}.csv",
        repo_type="dataset",
    )
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ui = row.get("user_input") or ""
            if not ui.strip():
                continue
            records.append({
                "user_input": ui[: args.max_chars],
                "toxicity": int(row.get("toxicity", 0) or 0),
                "jailbreaking": int(row.get("jailbreaking", 0) or 0),
            })
    print(f"ToxicChat {args.split}: {len(records)} records")

    if args.samples > 0 and len(records) > args.samples:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: args.samples]
        print(f"Sampled: {len(records)}")

    tox = sum(r["toxicity"] for r in records)
    jb = sum(r["jailbreaking"] for r in records)
    print(f"  toxicity=1: {tox} ({100*tox/len(records):.1f}%)")
    print(f"  jailbreaking=1: {jb} ({100*jb/len(records):.1f}%)")

    print(f"\nModel:   {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print("Loading model...")
    model = GLiNER2.from_pretrained(args.model)
    if args.adapter:
        model.load_adapter(args.adapter)
        if hasattr(model, "merge_lora"):
            model.merge_lora()
    print("Loaded.")

    schema = (model.create_schema()
        .classification(task="safety", labels=SAFETY_LABELS)
        .classification(task="adversarial", labels=ADVERSARIAL_LABELS, multi_label=True)
        .classification(task="harmful", labels=HARMFUL_LABELS, multi_label=True)
    )

    safety_tp = safety_fp = safety_fn = safety_tn = 0
    adv_tp = adv_fp = adv_fn = adv_tn = 0
    harm_tp = harm_fp = harm_fn = harm_tn = 0

    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            result = model.extract(rec["user_input"], schema=schema)
        except Exception:
            result = {}

        gt_unsafe = rec["toxicity"] == 1
        gt_adv = rec["jailbreaking"] == 1
        gt_harm = rec["toxicity"] == 1

        pred_safety = result.get("safety", "safe")
        pred_unsafe = pred_safety == "unsafe"
        pred_adv_labels = set(result.get("adversarial", ["none"])) - {"none"}
        pred_harm_labels = set(result.get("harmful", ["none"])) - {"none"}
        pred_adv = len(pred_adv_labels) > 0
        pred_harm = len(pred_harm_labels) > 0

        # safety as binary (unsafe = positive class)
        if gt_unsafe and pred_unsafe:
            safety_tp += 1
        elif gt_unsafe and not pred_unsafe:
            safety_fn += 1
        elif not gt_unsafe and pred_unsafe:
            safety_fp += 1
        else:
            safety_tn += 1

        if gt_adv and pred_adv:
            adv_tp += 1
        elif gt_adv and not pred_adv:
            adv_fn += 1
        elif not gt_adv and pred_adv:
            adv_fp += 1
        else:
            adv_tn += 1

        if gt_harm and pred_harm:
            harm_tp += 1
        elif gt_harm and not pred_harm:
            harm_fn += 1
        elif not gt_harm and pred_harm:
            harm_fp += 1
        else:
            harm_tn += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(records)}] {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    n = len(records)

    safety_acc = (safety_tp + safety_tn) / n
    sp, sr, sf = binary_f1(safety_tp, safety_fp, safety_fn)
    ap, ar, af = binary_f1(adv_tp, adv_fp, adv_fn)
    hp, hr, hf = binary_f1(harm_tp, harm_fp, harm_fn)

    print(f"\n{'='*70}")
    print(f"TOXICCHAT {args.split.upper()} ({n} samples, {elapsed:.1f}s, {1000*elapsed/n:.0f}ms/rec)")
    print(f"{'='*70}")
    print(f"  Safety  accuracy: {safety_acc:.4f}")
    print(f"    unsafe  P/R/F1: {sp} / {sr} / {sf}")
    print(f"    TP={safety_tp} FP={safety_fp} FN={safety_fn} TN={safety_tn}")
    print(f"  Adv     binary  F1: {af}   P={ap}  R={ar}")
    print(f"    TP={adv_tp} FP={adv_fp} FN={adv_fn} TN={adv_tn}")
    print(f"  Harm    binary  F1: {hf}   P={hp}  R={hr}")
    print(f"    TP={harm_tp} FP={harm_fp} FN={harm_fn} TN={harm_tn}")
    print(f"{'='*70}")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": f"lmsys/toxic-chat/{args.split}",
        "samples": n,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / n),
        "safety": {
            "accuracy": round(safety_acc, 4),
            "unsafe_precision": sp, "unsafe_recall": sr, "unsafe_f1": sf,
            "tp": safety_tp, "fp": safety_fp, "fn": safety_fn, "tn": safety_tn,
        },
        "adversarial": {
            "binary_f1": af, "binary_precision": ap, "binary_recall": ar,
            "tp": adv_tp, "fp": adv_fp, "fn": adv_fn, "tn": adv_tn,
        },
        "harmful": {
            "binary_f1": hf, "binary_precision": hp, "binary_recall": hr,
            "tp": harm_tp, "fp": harm_fp, "fn": harm_fn, "tn": harm_tn,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
