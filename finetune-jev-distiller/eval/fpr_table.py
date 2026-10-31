"""Per-dataset false-block rate and recall on GuardRate items unseen in training (0.5 threshold).

    python fpr_table.py run5=runs/d3-ft/guardrate_en.jsonl run6=runs/d3-v6/guardrate_en.jsonl
"""
import json, sys
from collections import defaultdict
items = {it["id"]: it for it in map(json.loads, open("data/guardrate_en.jsonl"))}
seen = set(map(int, json.load(open("data/guardrate_en_seen_r5r6.json"))["flagged"]))
S = dict(a.split("=") for a in sys.argv[1:])
P = {k: {r["id"]: r["p_unsafe"] for r in map(json.loads, open(v))} for k, v in S.items()}
agg = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in S}  # fp, neg, tp, pos
for i, it in items.items():
    if i in seen:
        continue
    for k in S:
        a = agg[k][it["dataset"]]
        b = P[k][i] >= 0.5
        if it["unsafe"]:
            a[2] += b; a[3] += 1
        else:
            a[0] += b; a[1] += 1
ds = sorted(items[i]["dataset"] for i in items)
ds = sorted(set(ds))
print(f"{'unseen items, FPR% / recall':32s}" + "".join(f"{k:>16s}" for k in S))
tot = {k: [0, 0, 0, 0] for k in S}
for d in ds:
    row = ""
    for k in S:
        fp, n, tp, p = agg[k][d]
        for j, x in enumerate((fp, n, tp, p)): tot[k][j] += x
        row += f"{(f'{100*fp/n:5.1f}' if n else '    -'):>8}/{(f'{tp/p:.3f}' if p else '  -  '):>7}"
    print(f"{d:32s}{row}")
print(f"{'ALL (pooled)':32s}" + "".join(f"{100*t[0]/t[1]:8.1f}/{t[2]/t[3]:7.3f}" for t in tot.values()))
