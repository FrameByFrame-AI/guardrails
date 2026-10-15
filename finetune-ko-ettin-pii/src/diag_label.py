#!/usr/bin/env python3
"""Bucketed error diagnosis for any label(s): pass labels as argv."""
import json, os, sys
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = os.environ.get("KOE_DIAG_MODEL", "/models/outputs/ko-ettin-encoder-150m-ko-pii")
DATA = "/gen/ko_pii_opf_v4/test.jsonl"
TARGETS = sys.argv[1:] or ["private_person", "personal_handle"]

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


def classify(g, p):
    if g == p: return "EXACT"
    if g[0] == p[0]: return "RIGHT_EDGE_OFF"
    if g[1] == p[1]: return "LEFT_EDGE_OFF"
    if max(g[0], p[0]) < min(g[1], p[1]): return "OVERLAP"
    return "DISJOINT"


for TARGET in TARGETS:
    tp = fp = fn = 0
    buckets, mislabel = {}, {}
    shown = 0
    print(f"\n########## {TARGET} ##########")
    for line in open(DATA):
        r = json.loads(line); text = r["text"]
        gold = [(s["start"], s["end"]) for s in (r.get("label") or []) if s["category"] == TARGET]
        spans, toks, off = decode_spans(text)
        pred = [(s, e) for cat, s, e in spans if cat == TARGET]
        gset, pset = set(gold), set(pred)
        tp += len(gset & pset); fp += len(pset - gset); fn += len(gset - pset)
        for g in gold:
            if g in pset: continue
            # overlapping pred of ANY label (to detect label confusion)
            best = None; best_lab = None
            for cat, s, e in spans:
                if max(g[0], s) < min(g[1], e): best = (s, e); best_lab = cat; break
            if best is None:
                buckets["MISSED"] = buckets.get("MISSED", 0) + 1
                if shown < 16:
                    shown += 1
                    print(f"[MISSED] gold={text[g[0]:g[1]]!r}  ctx={text[max(0,g[0]-8):g[1]+8]!r}")
                continue
            if best_lab != TARGET:
                key = f"WRONG_LABEL->{best_lab}"
                buckets[key] = buckets.get(key, 0) + 1
                if shown < 16:
                    shown += 1
                    print(f"[{key}] gold={text[g[0]:g[1]]!r}  pred={text[best[0]:best[1]]!r}")
                continue
            kind = classify(g, best)
            buckets[kind] = buckets.get(kind, 0) + 1
            if shown < 16:
                shown += 1
                frag = [t+('*' if t in added else '') for t, o in zip(toks, off)
                        if o[1] > min(g[0], best[0]) - 1 and o[0] < max(g[1], best[1]) + 1 and o[1] > o[0]]
                print(f"[{kind}] gold={text[g[0]:g[1]]!r}  pred={text[best[0]:best[1]]!r}  toks={frag}")
    print(f"--- {TARGET}: TP={tp} FP={fp} FN={fn} "
          f"P={tp/(tp+fp) if tp+fp else 0:.3f} R={tp/(tp+fn) if tp+fn else 0:.3f}")
    print("    buckets:", buckets)
