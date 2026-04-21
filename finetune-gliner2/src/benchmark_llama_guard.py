#!/usr/bin/env python3
"""
Benchmark Llama Guard 4 (RedHatAI FP8-dynamic) on the GLiNER2 guardrail test set.

FP8 dynamic quantization on Ampere (A5000) GPUs isn't natively supported, and
vLLM's Marlin kernel emulation produces garbage output on this model. We use
transformers directly — compressed-tensors dequantizes FP8 weights to bf16 at
load time, so inference is correct (just slower than native FP8).

Outputs safety + harmful only. Adversarial and entities are reported as
"not supported by Llama Guard 4".

Usage:
    python benchmark_llama_guard.py \
        --model-path /models/RedHatAI-Llama-Guard-4-12B-FP8-dynamic \
        --test-data /data/guardrail_gliner2_test.jsonl \
        --samples 2000 \
        --batch-size 4 \
        --output /data/benchmark_results/llama_guard_4_benchmark.json
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration


LG4_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex Crimes",
    "S4": "Child Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

LG4_TO_HARMFUL = {
    "S1": "violence",
    "S2": "criminal_planning",
    "S3": "sexual_content",
    "S4": "child_exploitation",
    "S5": "misinformation",
    "S6": "unauthorized_advice",
    "S7": "fraud",
    "S8": "copyright",
    "S9": "weapons",
    "S10": "hate_speech",
    "S11": "self_harm",
    "S12": "sexual_content",
    "S13": "misinformation",
    "S14": "malware",
}


def f1_score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return round(2 * p * r / (p + r), 4) if (p + r) else 0


def precision(tp, fp):
    return round(tp / (tp + fp), 4) if (tp + fp) else 0


def recall(tp, fn):
    return round(tp / (tp + fn), 4) if (tp + fn) else 0


def parse_lg4_response(text):
    t = text.strip()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines:
        return True, []
    first = lines[0].lower()
    if first.startswith("safe"):
        return True, []
    if first.startswith("unsafe"):
        codes = []
        if len(lines) > 1:
            raw = lines[1].replace(" ", "")
            for c in raw.split(","):
                c = c.strip().upper()
                if c in LG4_CATEGORIES:
                    codes.append(c)
        return False, codes
    return True, []


def classify_batch(model, processor, texts, device, max_new_tokens=32):
    """Run Llama Guard 4 on a batch of input texts."""
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": t}]}]
        for t in texts
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=False,
        )

    gen_tokens = output[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/RedHatAI-Llama-Guard-4-12B-FP8-dynamic")
    parser.add_argument("--test-data", default="/data/guardrail_gliner2_test.jsonl")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="/data/benchmark_results/llama_guard_4_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.test_data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.samples > 0 and len(records) > args.samples:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[:args.samples]

    print(f"Model: {args.model_path}")
    print(f"Test records: {len(records)}")
    print(f"Batch size:   {args.batch_size}")

    print("Loading model (FP8 → bf16 dequant)...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    model = Llama4ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Loaded on {device}.")

    outputs = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        texts = [r["input"] for r in batch]
        try:
            decoded = classify_batch(model, processor, texts, device)
        except Exception as e:
            print(f"  batch error: {e}", flush=True)
            decoded = [""] * len(batch)
        outputs.extend(decoded)
        if (i + len(batch)) % 50 == 0 or (i + len(batch)) == len(records):
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(records)}] {elapsed:.1f}s  {rate:.2f} rec/s", flush=True)

    elapsed = time.time() - t0

    gt_safety, pred_safety = [], []
    gt_harmful, pred_harmful = [], []
    gt_adversarial = []

    for rec, raw in zip(records, outputs):
        for cls in rec["output"]["classifications"]:
            if cls["task"] == "safety":
                gt_safety.append(cls["true_label"])
            elif cls["task"] == "harmful":
                gt_harmful.append(cls["true_label"])
            elif cls["task"] == "adversarial":
                gt_adversarial.append(cls["true_label"])
        safe, codes = parse_lg4_response(raw)
        pred_safety.append("safe" if safe else "unsafe")
        mapped = list({LG4_TO_HARMFUL[c] for c in codes if c in LG4_TO_HARMFUL})
        pred_harmful.append(mapped if mapped else ["none"])

    correct = sum(1 for g, p in zip(gt_safety, pred_safety) if g == p)
    safety_accuracy = correct / len(gt_safety) if gt_safety else 0
    tp = sum(1 for g, p in zip(gt_safety, pred_safety) if g == "unsafe" and p == "unsafe")
    fp = sum(1 for g, p in zip(gt_safety, pred_safety) if g == "safe" and p == "unsafe")
    fn = sum(1 for g, p in zip(gt_safety, pred_safety) if g == "unsafe" and p == "safe")
    tn = sum(1 for g, p in zip(gt_safety, pred_safety) if g == "safe" and p == "safe")

    safety_result = {
        "accuracy": round(safety_accuracy, 4),
        "f1": f1_score(tp, fp, fn),
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    binary_tp = binary_fp = binary_tn = binary_fn = 0
    for gt_labels, pred_labels in zip(gt_harmful, pred_harmful):
        gt_set = set(gt_labels) - {"none"}
        pred_set = set(pred_labels) - {"none"}
        for lbl in gt_set & pred_set:
            per_label[lbl]["tp"] += 1
        for lbl in gt_set - pred_set:
            per_label[lbl]["fn"] += 1
        for lbl in pred_set - gt_set:
            per_label[lbl]["fp"] += 1
        if gt_set and pred_set:
            binary_tp += 1
        elif gt_set and not pred_set:
            binary_fn += 1
        elif not gt_set and pred_set:
            binary_fp += 1
        else:
            binary_tn += 1

    harmful_result = {
        "binary_f1": f1_score(binary_tp, binary_fp, binary_fn),
        "binary_precision": precision(binary_tp, binary_fp),
        "binary_recall": recall(binary_tp, binary_fn),
        "binary_tp": binary_tp, "binary_fp": binary_fp,
        "binary_tn": binary_tn, "binary_fn": binary_fn,
        "per_label": {k: dict(v) for k, v in per_label.items()},
    }

    # Binary adversarial F1: LG4 has no jailbreak / prompt-injection categories,
    # so we reduce to "did LG4 flag an adversarial-GT sample as unsafe?" using
    # the same pred_safety signal. No per-label breakdown is possible.
    adv_tp = adv_fp = adv_tn = adv_fn = 0
    for gt_labels, pred in zip(gt_adversarial, pred_safety):
        gt_pos = bool(set(gt_labels) - {"none"})
        pred_pos = pred == "unsafe"
        if gt_pos and pred_pos:
            adv_tp += 1
        elif gt_pos and not pred_pos:
            adv_fn += 1
        elif not gt_pos and pred_pos:
            adv_fp += 1
        else:
            adv_tn += 1

    adversarial_result = {
        "note": (
            "LG4 has no adversarial categories; binary F1 computed by treating "
            "any LG4 'unsafe' verdict as a positive adversarial prediction."
        ),
        "binary_f1": f1_score(adv_tp, adv_fp, adv_fn),
        "binary_precision": precision(adv_tp, adv_fp),
        "binary_recall": recall(adv_tp, adv_fn),
        "binary_tp": adv_tp, "binary_fp": adv_fp,
        "binary_tn": adv_tn, "binary_fn": adv_fn,
    }

    print(f"\n{'='*70}")
    print(f"LLAMA-GUARD-4 BENCHMARK ({len(records)} samples, {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Safety accuracy:           {safety_result['accuracy']}")
    print(f"  Safety F1 (unsafe-pos):    {safety_result['f1']}")
    print(f"    precision:               {safety_result['precision']}")
    print(f"    recall:                  {safety_result['recall']}")
    print(f"  Harmful binary F1:         {harmful_result['binary_f1']}")
    print(f"    precision:               {harmful_result['binary_precision']}")
    print(f"    recall:                  {harmful_result['binary_recall']}")
    print(f"  Adversarial binary F1:     {adversarial_result['binary_f1']}  "
          f"(LG4 'unsafe' ≈ positive)")
    print(f"    precision:               {adversarial_result['binary_precision']}")
    print(f"    recall:                  {adversarial_result['binary_recall']}")
    print(f"  Entities:                  not supported by Llama Guard 4")
    print(f"  Latency:                   {1000*elapsed/len(records):.0f}ms/record")
    print(f"{'='*70}")

    print(f"\n  Harmful per-label:")
    for lbl in sorted(harmful_result["per_label"],
                      key=lambda l: -(harmful_result["per_label"][l]["tp"] +
                                      harmful_result["per_label"][l]["fn"])):
        s = harmful_result["per_label"][lbl]
        print(f"    {lbl:<25} TP={s['tp']:>4} FP={s['fp']:>4} FN={s['fn']:>4} "
              f"F1={f1_score(s['tp'], s['fp'], s['fn'])}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model_path,
        "backend": "transformers (FP8 dequantized to bf16)",
        "samples": len(records),
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / len(records)),
        "batch_size": args.batch_size,
        "safety": safety_result,
        "harmful": harmful_result,
        "adversarial": adversarial_result,
        "entities": "not supported by Llama Guard 4",
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
