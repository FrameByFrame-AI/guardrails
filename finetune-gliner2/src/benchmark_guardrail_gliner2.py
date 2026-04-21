#!/usr/bin/env python3
"""
Benchmark GLiNER2 guardrail model on multi-task test data.

Evaluates all 5 classification tasks + entity extraction using the same
schema and labels as training. Run before AND after training for A/B comparison.

Usage:
    python benchmark_guardrail_gliner2.py \
        --model fastino/gliner2-multi-v1 \
        --test-data /data/guardrail_gliner2_test.jsonl \
        --samples 2000 \
        --output /data/benchmark_results/guardrail_gliner2_baseline.json

    # After training, with adapter:
    python benchmark_guardrail_gliner2.py \
        --model fastino/gliner2-multi-v1 \
        --adapter /models/guardrail-gliner2-trained \
        --test-data /data/guardrail_gliner2_test.jsonl
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from gliner2 import GLiNER2


# Must match training format exactly
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


def f1_score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return round(2 * p * r / (p + r), 4) if (p + r) else 0


def precision(tp, fp):
    return round(tp / (tp + fp), 4) if (tp + fp) else 0


def recall(tp, fn):
    return round(tp / (tp + fn), 4) if (tp + fn) else 0


def eval_single_label(gt_list, pred_list):
    """Evaluate single-label classification (safety)."""
    tp = fp = fn = tn = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gt, pred in zip(gt_list, pred_list):
        if gt == pred:
            per_label[gt]["tp"] += 1
        else:
            per_label[gt]["fn"] += 1
            per_label[pred]["fp"] += 1

    correct = sum(1 for g, p in zip(gt_list, pred_list) if g == p)
    accuracy = correct / len(gt_list) if gt_list else 0
    return {"accuracy": round(accuracy, 4), "per_label": dict(per_label)}


def eval_multi_label(gt_list, pred_list):
    """Evaluate multi-label classification (adversarial, harmful)."""
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for gt_labels, pred_labels in zip(gt_list, pred_list):
        gt_set = set(gt_labels) - {"none"}
        pred_set = set(pred_labels) - {"none"}

        for lbl in gt_set & pred_set:
            per_label[lbl]["tp"] += 1
        for lbl in gt_set - pred_set:
            per_label[lbl]["fn"] += 1
        for lbl in pred_set - gt_set:
            per_label[lbl]["fp"] += 1

    # Binary: did we detect ANY label (excluding "none")?
    binary_tp = binary_fp = binary_tn = binary_fn = 0
    for gt_labels, pred_labels in zip(gt_list, pred_list):
        gt_has = set(gt_labels) - {"none"}
        pred_has = set(pred_labels) - {"none"}
        if gt_has and pred_has:
            binary_tp += 1
        elif gt_has and not pred_has:
            binary_fn += 1
        elif not gt_has and pred_has:
            binary_fp += 1
        else:
            binary_tn += 1

    return {
        "binary_f1": f1_score(binary_tp, binary_fp, binary_fn),
        "binary_precision": precision(binary_tp, binary_fp),
        "binary_recall": recall(binary_tp, binary_fn),
        "binary_tp": binary_tp, "binary_fp": binary_fp,
        "binary_tn": binary_tn, "binary_fn": binary_fn,
        "per_label": {k: dict(v) for k, v in per_label.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--test-data", default="/data/guardrail_gliner2_test.jsonl")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", default="/data/benchmark_results/guardrail_gliner2_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load test records
    records = []
    with open(args.test_data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.samples > 0 and len(records) > args.samples:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[:args.samples]

    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Test records: {len(records)}")

    # Load model
    print("Loading model...")
    model = GLiNER2.from_pretrained(args.model)
    if args.adapter:
        model.load_adapter(args.adapter)
        if hasattr(model, "merge_lora"):
            model.merge_lora()
    print("Loaded.")

    # Build schema (guard-only — no entity head)
    schema = (model.create_schema()
        .classification(task="safety", labels=SAFETY_LABELS)
        .classification(task="adversarial", labels=ADVERSARIAL_LABELS, multi_label=True)
        .classification(task="harmful", labels=HARMFUL_LABELS, multi_label=True)
    )

    # Collect predictions
    gt_safety, pred_safety = [], []
    gt_adversarial, pred_adversarial = [], []
    gt_harmful, pred_harmful = [], []

    t0 = time.time()
    for i, rec in enumerate(records):
        # Ground truth
        for cls in rec["output"]["classifications"]:
            if cls["task"] == "safety":
                gt_safety.append(cls["true_label"])
            elif cls["task"] == "adversarial":
                gt_adversarial.append(cls["true_label"])
            elif cls["task"] == "harmful":
                gt_harmful.append(cls["true_label"])

        # Predict
        try:
            result = model.extract(rec["input"], schema=schema)
        except Exception:
            result = {}

        pred_safety.append(result.get("safety", "safe"))
        pred_adversarial.append(result.get("adversarial", ["none"]))
        pred_harmful.append(result.get("harmful", ["none"]))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(records)}] {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0

    # Evaluate
    safety_results = eval_single_label(gt_safety, pred_safety)
    adversarial_results = eval_multi_label(gt_adversarial, pred_adversarial)
    harmful_results = eval_multi_label(gt_harmful, pred_harmful)

    # Print summary
    print(f"\n{'='*70}")
    print(f"GUARDRAIL-GLINER2 BENCHMARK ({len(records)} samples, {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Safety accuracy:           {safety_results['accuracy']}")
    print(f"  Adversarial binary F1:     {adversarial_results['binary_f1']}")
    print(f"    precision:               {adversarial_results['binary_precision']}")
    print(f"    recall:                  {adversarial_results['binary_recall']}")
    print(f"  Harmful binary F1:         {harmful_results['binary_f1']}")
    print(f"    precision:               {harmful_results['binary_precision']}")
    print(f"    recall:                  {harmful_results['binary_recall']}")
    print(f"  Latency:                   {1000*elapsed/len(records):.0f}ms/record")
    print(f"{'='*70}")

    # Per-label adversarial
    print(f"\n  Adversarial per-label:")
    for lbl in sorted(adversarial_results["per_label"],
                      key=lambda l: -(adversarial_results["per_label"][l]["tp"] +
                                      adversarial_results["per_label"][l]["fn"])):
        s = adversarial_results["per_label"][lbl]
        print(f"    {lbl:<25} TP={s['tp']:>4} FP={s['fp']:>4} FN={s['fn']:>4} "
              f"F1={f1_score(s['tp'], s['fp'], s['fn'])}")

    # Per-label harmful
    print(f"\n  Harmful per-label:")
    for lbl in sorted(harmful_results["per_label"],
                      key=lambda l: -(harmful_results["per_label"][l]["tp"] +
                                      harmful_results["per_label"][l]["fn"])):
        s = harmful_results["per_label"][lbl]
        print(f"    {lbl:<25} TP={s['tp']:>4} FP={s['fp']:>4} FN={s['fn']:>4} "
              f"F1={f1_score(s['tp'], s['fp'], s['fn'])}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "samples": len(records),
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / len(records)),
        "safety": safety_results,
        "adversarial": adversarial_results,
        "harmful": harmful_results,
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
