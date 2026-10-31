"""Benchmark v2 per group: recall / false-block rate at 0.5 and AUC, AccuKnox Guard vs Jev and XGuard.

    python bench_v2_compare.py > runs/bench_v2_compare.txt

One-off analysis for experiments.md ("XGuard on our injection benchmark"). Benchmark v2 is data/benchmark.jsonl;
predictions: run 5 (runs/d3-ft/benchmark.jsonl), Jev (data/jev_labels.jsonl), XGuard 0.6B and 8B
(runs/{xguard-0.6b,xguard}/benchmark_v2.jsonl, guardrate.py output by item id). A group without unsafe or without
safe items shows "-" for the missing rate and AUC.
"""

import json

from card_results import auc


def by_text(path, key):
    return {r["text"]: key(r) for r in map(json.loads, open(path))}


def main():
    bench = [json.loads(l) for l in open("data/benchmark.jsonl")]
    preds = {"AccuKnox Guard": by_text("runs/d3-ft/benchmark.jsonl", lambda r: r["block_prob"]),
             "Jev": {r["text"]: r["jev"]["block_prob"] for r in map(json.loads, open("data/jev_labels.jsonl"))
                     if r.get("jev")}}
    for name, run in (("XGuard-0.6B", "xguard-0.6b"), ("XGuard-8B", "xguard")):
        rows = {r["id"]: r["p_unsafe"] for r in map(json.loads, open(f"runs/{run}/benchmark_v2.jsonl"))}
        preds[name] = {bench[i]["text"]: p for i, p in rows.items()}

    print("coverage:", {n: sum(b["text"] in p for b in bench) for n, p in preds.items()})
    print(f"{'group':20s} " + " ".join(f"{n:>24s}" for n in preds))
    for group in sorted({b["group"] for b in bench}) + ["ALL"]:
        rows = [b for b in bench if group == "ALL" or b["group"] == group]
        cells = []
        for p in preds.values():
            pos = [p[b["text"]] for b in rows if b["gold_block"] and b["text"] in p]
            neg = [p[b["text"]] for b in rows if not b["gold_block"] and b["text"] in p]
            rec = f"{sum(x >= .5 for x in pos) / len(pos):.2f}" if pos else "-"
            fbr = f"{sum(x >= .5 for x in neg) / len(neg):.2f}" if neg else "-"
            a = f"{auc(pos, neg):.3f}" if pos and neg else "-"
            cells.append(f"R{rec} FB{fbr} A{a}")
        print(f"{group:20s} " + " ".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    main()
