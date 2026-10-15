#!/usr/bin/env python3
"""Diagnose private_address errors: gold vs predicted spans + tokenization."""
import json, sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "/models/outputs/ko-ettin-encoder-150m-ko-pii"
DATA = "/gen/ko_pii_opf_v4/test.jsonl"
TARGET = "private_address"

tok = AutoTokenizer.from_pretrained(MODEL)
added = set(tok.get_added_vocab().keys())
model = AutoModelForTokenClassification.from_pretrained(MODEL, torch_dtype="auto")
if hasattr(model.config, "reference_compile"):
    model.config.reference_compile = False
model.eval().cuda()
id2label = {int(k): v for k, v in model.config.id2label.items()}


def decode_spans(text):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=256, return_tensors="pt")
    off = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        pred = model(**{k: v.cuda() for k, v in enc.items()}).logits[0].argmax(-1).tolist()
    spans, cur, st, prev_e = [], None, None, 0
    for i, pid in enumerate(pred):
        lab = id2label[pid]; s, e = off[i]
        if s == e:
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
    if cur: spans.append((cur, st, prev_e))
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    return spans, toks, off


def classify(gold, pred):
    if gold == pred: return "EXACT"
    g0, g1 = gold; p0, p1 = pred
    if g0 == p0 and g1 != p1: return "RIGHT_EDGE_OFF"
    if g1 == p1 and g0 != p0: return "LEFT_EDGE_OFF"
    if max(g0, p0) < min(g1, p1): return "OVERLAP"
    return "DISJOINT"

tp = fp = fn = 0
buckets = {}
shown = 0
for line in open(DATA):
    r = json.loads(line); text = r["text"]
    gold = [(s["start"], s["end"]) for s in (r.get("label") or []) if s["category"] == TARGET]
    spans, toks, off = decode_spans(text)
    pred = [(s, e) for cat, s, e in spans if cat == TARGET]
    gset, pset = set(gold), set(pred)
    tp += len(gset & pset); fp += len(pset - gset); fn += len(gset - pset)
    # classify each gold vs best-overlapping pred
    for g in gold:
        if g in pset: continue
        best = None
        for p in pred:
            if max(g[0], p[0]) < min(g[1], p[1]): best = p; break
        kind = classify(g, best) if best else "MISSED"
        buckets[kind] = buckets.get(kind, 0) + 1
        if shown < 22 and best:
            shown += 1
            gs, ge = g
            frag = [t for t, o in zip(toks, off) if o[1] > min(gs, best[0]) - 1 and o[0] < max(ge, best[1]) + 1 and o[1] > o[0]]
            print(f"[{kind}] GOLD={text[gs:ge]!r}  PRED={text[best[0]:best[1]]!r}")
            print(f"        ctx={text[max(0,gs-8):ge+10]!r}")
            print(f"        toks={[ (t+('*' if t in added else '')) for t in frag]}")

print(f"\n=== {TARGET}: TP={tp} FP={fp} FN={fn} "
      f"P={tp/(tp+fp) if tp+fp else 0:.3f} R={tp/(tp+fn) if tp+fn else 0:.3f} ===")
print("error buckets (gold not matched exactly):", buckets)
