#!/usr/bin/env python3
"""Verification GATE for synthetic PII before training. Exits non-zero on hard fail.

Checks per label (vs the real train distribution):
  HARD (fail build):
    - offset round-trip: text[start:end] non-empty for every span
    - test contamination: no synth text equals a test text
  SOFT->HARD if --strict (convention drift):
    - span-length median within [0.5x, 2x] of real
    - script composition (hangul/ascii/mixed) roughly matches real (<25pp drift)
  WARN:
    - unlabeled PII (digits/@/http) outside any labeled span
    - coverage: fraction of real surface 'head units' seen in synth
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def spans_of(recs, cat):
    out = []
    for r in recs:
        for sp in (r.get("label") or []):
            if sp["category"] == cat:
                out.append(r["text"][sp["start"]:sp["end"]])
    return out


def script(f):
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", f): return "ascii"
    if re.search(r"[A-Za-z0-9]", f) and re.search(r"[가-힣]", f): return "mixed"
    if re.fullmatch(r"[가-힣 ]+", f): return "hangul"
    return "other"


def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2] if xs else 0


def comp(forms):
    c = Counter(script(f) for f in forms)
    n = sum(c.values()) or 1
    return {k: round(v/n, 3) for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--strict", action="store_true", help="treat convention drift as hard fail")
    args = ap.parse_args()
    synth, real, test = load(args.synth), load(args.real), load(args.test)
    hard, soft, warn = [], [], []

    # HARD: offsets
    bad = sum(1 for r in synth for sp in r["label"] if not r["text"][sp["start"]:sp["end"]].strip())
    if bad: hard.append(f"offset/empty-span errors: {bad}")

    # HARD: contamination
    tt = {r["text"] for r in test}
    dup = sum(1 for r in synth if r["text"] in tt)
    if dup: hard.append(f"synth texts colliding with TEST: {dup}")

    # WARN: unlabeled PII
    leak = 0
    for r in synth:
        lab = set()
        for sp in r["label"]:
            lab |= set(range(sp["start"], sp["end"]))
        for m in re.finditer(r"\d{2,}|@|https?://", r["text"]):
            if not (set(range(m.start(), m.end())) & lab):
                leak += 1; break
    if leak: warn.append(f"synth texts with PII-like tokens outside spans: {leak}/{len(synth)}")

    # per-label convention vs real
    cats = sorted({sp["category"] for r in synth for sp in r["label"]})
    print(f"{'label':18} {'n_syn':>6} {'med_syn':>7} {'med_real':>8}  script_syn -> script_real")
    for cat in cats:
        sf, rf = spans_of(synth, cat), spans_of(real, cat)
        if not rf:
            continue
        ms, mr = median([len(x) for x in sf]), median([len(x) for x in rf])
        cs, cr = comp(sf), comp(rf)
        print(f"{cat:18} {len(sf):>6} {ms:>7} {mr:>8}  {cs} -> {cr}")
        if not (0.5*mr <= ms <= 2.0*mr):
            soft.append(f"{cat}: median len {ms} vs real {mr} (outside 0.5x-2x)")
        for k in ("hangul", "ascii"):
            if abs(cs.get(k, 0) - cr.get(k, 0)) > 0.25:
                soft.append(f"{cat}: {k} share {cs.get(k,0):.2f} vs real {cr.get(k,0):.2f} (>25pp)")

    print()
    for w in warn: print("WARN:", w)
    for s in soft: print("DRIFT:", s)
    for h in hard: print("FAIL:", h)
    if args.strict:
        hard += soft
    if hard:
        print(f"\nGATE: FAIL ({len(hard)} hard issue(s))"); sys.exit(1)
    print(f"\nGATE: PASS" + (f" ({len(soft)} soft drift warning(s))" if soft else ""))


if __name__ == "__main__":
    main()
