#!/usr/bin/env python3
"""
Benchmark Llama Guard 4 (RedHatAI FP8-dynamic) on ToxicChat (lmsys/toxic-chat).

Uses transformers (FP8 -> bf16 dequant) same as benchmark_llama_guard.py.
LG4 outputs safe/unsafe + S-category codes. We map:
  - LG4 "unsafe" verdict  -> safety=unsafe AND harm positive AND (optionally) adv positive
  - LG4 has no adversarial categories, so adv binary F1 falls back to
    "did LG4 flag a jailbreak sample as unsafe?"
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from huggingface_hub import hf_hub_download


LG4_CATEGORIES = {f"S{i}" for i in range(1, 15)}


def parse_lg4_response(text):
    t = text.strip()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines:
        return True
    first = lines[0].lower()
    return not first.startswith("unsafe")


def binary_metrics(tp, fp, fn, tn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def classify_batch(model, processor, texts, device, max_new_tokens=24):
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
    return processor.batch_decode(gen_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/RedHatAI-Llama-Guard-4-12B-FP8-dynamic")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--output", default="/data/benchmark_results/toxicchat_llama_guard_4.json")
    parser.add_argument("--seed", type=int, default=42)
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

    tox = sum(r["toxicity"] for r in records)
    jb = sum(r["jailbreaking"] for r in records)
    print(f"Sampled: {len(records)}  (tox={tox}, jb={jb})")

    print(f"Model: {args.model_path}")
    print("Loading model (FP8 -> bf16 dequant)...")
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

    raw_outputs = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        texts = [r["user_input"] for r in batch]
        try:
            decoded = classify_batch(model, processor, texts, device)
        except Exception as e:
            print(f"  batch error: {e}", flush=True)
            decoded = [""] * len(batch)
        raw_outputs.extend(decoded)
        done = i + len(batch)
        if done % 50 == 0 or done == len(records):
            elapsed = time.time() - t0
            print(f"  [{done}/{len(records)}] {elapsed:.1f}s  {done/elapsed:.2f} rec/s", flush=True)

    elapsed = time.time() - t0
    n = len(records)

    safety_tp = safety_fp = safety_fn = safety_tn = 0
    adv_tp = adv_fp = adv_fn = adv_tn = 0
    harm_tp = harm_fp = harm_fn = harm_tn = 0

    for rec, raw in zip(records, raw_outputs):
        lg4_safe = parse_lg4_response(raw)
        pred_unsafe = not lg4_safe

        gt_unsafe = rec["toxicity"] == 1
        gt_adv = rec["jailbreaking"] == 1
        gt_harm = rec["toxicity"] == 1

        if gt_unsafe and pred_unsafe:     safety_tp += 1
        elif gt_unsafe and not pred_unsafe: safety_fn += 1
        elif not gt_unsafe and pred_unsafe: safety_fp += 1
        else:                              safety_tn += 1

        if gt_adv and pred_unsafe:        adv_tp += 1
        elif gt_adv and not pred_unsafe:  adv_fn += 1
        elif not gt_adv and pred_unsafe:  adv_fp += 1
        else:                             adv_tn += 1

        if gt_harm and pred_unsafe:       harm_tp += 1
        elif gt_harm and not pred_unsafe: harm_fn += 1
        elif not gt_harm and pred_unsafe: harm_fp += 1
        else:                             harm_tn += 1

    safety_acc = (safety_tp + safety_tn) / n
    sm = binary_metrics(safety_tp, safety_fp, safety_fn, safety_tn)
    am = binary_metrics(adv_tp, adv_fp, adv_fn, adv_tn)
    hm = binary_metrics(harm_tp, harm_fp, harm_fn, harm_tn)

    print(f"\n{'='*70}")
    print(f"LG4 on TOXICCHAT {args.split.upper()} ({n} samples, {elapsed:.1f}s, {1000*elapsed/n:.0f}ms/rec)")
    print(f"{'='*70}")
    print(f"  Safety  accuracy: {safety_acc:.4f}")
    print(f"    unsafe  P/R/F1: {sm['precision']} / {sm['recall']} / {sm['f1']}")
    print(f"  Adv     F1: {am['f1']}   P={am['precision']}  R={am['recall']}  (LG4 'unsafe' ~= adv pos)")
    print(f"  Harm    F1: {hm['f1']}   P={hm['precision']}  R={hm['recall']}")
    print(f"{'='*70}")

    summary = {
        "model": args.model_path,
        "backend": "transformers (FP8 dequantized to bf16)",
        "dataset": f"lmsys/toxic-chat/{args.split}",
        "samples": n,
        "elapsed_sec": round(elapsed, 1),
        "ms_per_record": round(1000 * elapsed / n),
        "batch_size": args.batch_size,
        "safety": {"accuracy": round(safety_acc, 4), **{f"unsafe_{k}": v for k, v in sm.items() if k in {"precision","recall","f1"}}, **{k: sm[k] for k in ["tp","fp","fn","tn"]}},
        "adversarial": {"binary_f1": am["f1"], "binary_precision": am["precision"], "binary_recall": am["recall"], **{k: am[k] for k in ["tp","fp","fn","tn"]},
                        "note": "LG4 has no adv category; uses safety verdict as proxy"},
        "harmful": {"binary_f1": hm["f1"], "binary_precision": hm["precision"], "binary_recall": hm["recall"], **{k: hm[k] for k in ["tp","fp","fn","tn"]}},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
