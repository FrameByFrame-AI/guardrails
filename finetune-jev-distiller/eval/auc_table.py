"""ROC AUC per mixed-label GuardRate dataset and pooled, on items unseen in training: ranking quality independent
of the threshold. Edit S to choose systems.

    python auc_table.py
"""
import json
items = {it["id"]: it for it in map(json.loads, open("data/guardrate_en.jsonl"))}
seen = set(map(int, json.load(open("data/guardrate_en_seen_r5r6.json"))["flagged"]))
S = {"untrained": "runs/d3/guardrate_en.jsonl", "run5": "runs/d3-ft/guardrate_en.jsonl",
     "run6": "runs/d3-v6/guardrate_en.jsonl", "run7": "runs/d3-v7/guardrate_en.jsonl", "run8": "runs/d3-v8/guardrate_en.jsonl", "xguard8b": "runs/xguard/guardrate_en.jsonl",
     "shieldstral": "runs/shieldstral/guardrate_en.jsonl"}
def auc(pos, neg):
    allv = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    # rank-sum with ties averaged
    r, i, ranks = 0, 0, []
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]: j += 1
        for k in range(i, j): ranks.append(((i + j + 1) / 2, allv[k][1]))
        i = j
    rp = sum(rk for rk, y in ranks if y)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
mixed = sorted({it["dataset"] for it in items.values() if it["unsafe"]} & {it["dataset"] for it in items.values() if not it["unsafe"]})
print(f"{'unseen items, ROC AUC':26s}" + "".join(f"{k:>12s}" for k in S))
P = {k: {r["id"]: r["p_unsafe"] for r in map(json.loads, open(v))} for k, v in S.items()}
for d in mixed + ["POOLED"]:
    ids = [i for i, it in items.items() if i not in seen and (d == "POOLED" or it["dataset"] == d)]
    row = ""
    for k in S:
        pos = [P[k][i] for i in ids if items[i]["unsafe"]]; neg = [P[k][i] for i in ids if not items[i]["unsafe"]]
        row += f"{auc(pos, neg):12.3f}"
    print(f"{d:26s}{row}")
