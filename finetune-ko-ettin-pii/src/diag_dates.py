#!/usr/bin/env python3
"""Diagnose private_date errors: show gold vs predicted date spans for ko-ettin."""
import json, sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/models/outputs/ko-ettin-encoder-150m-ko-pii"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/gen/ko_pii_opf_v4/test.jsonl"
TARGET = "private_date"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL, torch_dtype="auto")
if hasattr(model.config, "reference_compile"):
    model.config.reference_compile = False
model.eval().cuda()
id2label = {int(k): v for k, v in model.config.id2label.items()}


def decode_spans(text):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=256, return_tensors="pt")
    off = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = model(**{k: v.cuda() for k, v in enc.items()}).logits[0]
    pred = logits.argmax(-1).tolist()
    spans, cur, st = [], None, None
    for i, pid in enumerate(pred):
        lab = id2label[pid]
        s, e = off[i]
        if s == e:  # special token
            if cur: spans.append((cur, st, prev_e)); cur = None
            continue
        if lab == "O":
            if cur: spans.append((cur, st, prev_e)); cur = None
        else:
            pre, cat = lab.split("-", 1)
            if pre in ("B", "S") or cat != cur:
                if cur: spans.append((cur, st, prev_e))
                cur, st = cat, s
            prev_e = e
        prev_e = e
    if cur: spans.append((cur, st, prev_e))
    return spans, [tok.convert_ids_to_tokens(i) for i in enc["input_ids"][0].tolist()], off


tp = fp = fn = 0
shown = 0
for line in open(DATA):
    r = json.loads(line)
    text = r["text"]
    gold = {(s["start"], s["end"]) for s in (r.get("label") or []) if s["category"] == TARGET}
    if not gold and TARGET not in text:  # skip irrelevant
        pass
    spans, toks, off = decode_spans(text)
    pred = {(s, e) for cat, s, e in spans if cat == TARGET}
    if not gold and not pred:
        continue
    tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    if gold != pred and shown < 25:
        shown += 1
        print(f"\nTEXT: {text!r}")
        print(f"  GOLD: {[text[s:e] for s,e in sorted(gold)]}")
        print(f"  PRED: {[(text[s:e], cat) for cat,s,e in spans if (s,e) in (pred|gold) or cat==TARGET]}")
        # show tokenization around the date
        if gold:
            gs = min(s for s, e in gold)
            frag = [(t, o) for t, o in zip(toks, off) if o[0] < max(e for s,e in gold)+3 and o[1] > gs-3 and o[1] > o[0]]
            print(f"  TOKENS@date: {[t for t,o in frag]}")

print(f"\n=== private_date: TP={tp} FP={fp} FN={fn} "
      f"P={tp/(tp+fp) if tp+fp else 0:.3f} R={tp/(tp+fn) if tp+fn else 0:.3f}===")
