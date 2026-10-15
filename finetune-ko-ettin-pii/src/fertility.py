#!/usr/bin/env python3
"""Tokenizer fertility: original Ettin vs ko-ettin expanded, on Korean PII sentences."""
import json, random
from transformers import AutoTokenizer

base = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-150m")
ko = AutoTokenizer.from_pretrained("/models/FrameByFrame/ettin-150m-korean-PII")

recs = []
for f in ["/gen/ko_pii_opf_v4/train.jsonl",
          "/gen/ko_pii_opf_v4/validation.jsonl",
          "/gen/ko_pii_opf_v4/test.jsonl"]:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if isinstance(r.get("text"), str) and r["text"].strip():
            recs.append(r)
random.Random(42).shuffle(recs)
recs = recs[:10000]
print(f"sentences: {len(recs)}")


def fert(tok):
    T = C = E = atok = an = 0
    for r in recs:
        t = r["text"]
        T += len(tok(t, add_special_tokens=False)["input_ids"])
        C += sum(1 for ch in t if not ch.isspace())
        E += len(t.split())
        for sp in (r.get("label") or []):
            if sp["category"] == "private_address":
                a = t[sp["start"]:sp["end"]]
                atok += len(tok(a, add_special_tokens=False)["input_ids"]); an += 1
    return T / E, T / C, atok / max(an, 1), an


be, bc, ba, n = fert(base)
ke, kc, ka, _ = fert(ko)
print(f"address spans measured: {n}")
print(f"{'metric':22} {'Ettin':>8} {'ko-ettin':>9} {'reduction':>10}")
for name, b, k in [("tokens/eojeol", be, ke), ("tokens/char", bc, kc), ("addr span tokens", ba, ka)]:
    print(f"{name:22} {b:>8.2f} {k:>9.2f} {(k-b)/b*100:>9.0f}%")
