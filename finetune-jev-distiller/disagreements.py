"""Sample GuardRate items where runs 5 and 6 are wrong and XGuard-8B right (and where all three are wrong).

    python disagreements.py aegis_2_0_prompt toxicchat
"""
import json, random, sys
from collections import Counter
items = {it["id"]: it for it in map(json.loads, open("data/guardrate_en.jsonl"))}
seen = set(map(int, json.load(open("data/guardrate_en_seen_r5r6.json"))["flagged"]))
L = lambda p: {r["id"]: r["p_unsafe"] for r in map(json.loads, open(p))}
r5, r6, xg = L("runs/d3-ft/guardrate_en.jsonl"), L("runs/d3-v6/guardrate_en.jsonl"), L("runs/xguard/guardrate_en.jsonl")
rng = random.Random(0)
for d in sys.argv[1:]:
    ids = [i for i, it in items.items() if it["dataset"] == d and i not in seen]
    # ours wrong (both runs), XGuard right, at 0.5
    fb = [i for i in ids if not items[i]["unsafe"] and r5[i] >= .5 and r6[i] >= .5 and xg[i] < .5]
    miss = [i for i in ids if items[i]["unsafe"] and r5[i] < .5 and r6[i] < .5 and xg[i] >= .5]
    allfb = [i for i in ids if not items[i]["unsafe"] and r5[i] >= .5 and r6[i] >= .5 and xg[i] >= .5]
    allmiss = [i for i in ids if items[i]["unsafe"] and r5[i] < .5 and r6[i] < .5 and xg[i] < .5]
    n = len(ids)
    print(f"\n##### {d}: {n} items | ours-only false blocks {len(fb)}, ours-only misses {len(miss)} | "
          f"all-three false blocks {len(allfb)}, all-three misses {len(allmiss)}")
    for name, lst in (("OURS-ONLY FALSE BLOCK (label safe)", fb), ("OURS-ONLY MISS (label unsafe)", miss),
                      ("ALL THREE BLOCK, label safe", allfb), ("ALL THREE ALLOW, label unsafe", allmiss)):
        print(f"--- {name}")
        for i in rng.sample(lst, min(6, len(lst))):
            it = items[i]
            t = it["prompt"].replace("\n", " ")[:170]
            rsp = (" || R: " + it["response"].replace("\n", " ")[:120]) if it.get("response") else ""
            print(f"  r5 {r5[i]:.2f} r6 {r6[i]:.2f} xg {xg[i]:.2f} | {t}{rsp}")
