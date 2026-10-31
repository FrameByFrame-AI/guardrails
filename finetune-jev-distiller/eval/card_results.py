"""The model card's results table: every system run on the GuardRate English rebuild, on items unseen in training.

    python card_results.py > runs/card_results.md

Columns: GuardRate integral (all 23 datasets, and core = without the 3 whose rebuild departs from the published
scores), ROC AUC pooled over all unseen items (as auc_table.py POOLED), p50 latency. Local models' latency comes
from runs/<m>/latency_a5000.jsonl (scripts/latency_a5000.sh: all on one RTX A5000, same 1,000-item sample), since
the full runs were split across two GPUs; hosted APIs' from their full runs (network included). Threshold 0.5
(Bedrock: its own decision; AUC from its filter confidence). Systems without predictions yet are skipped.
"""

import json
import os
import statistics

from guardrate import load_preds, score_items

SEEN = "data/guardrate_en_seen_r5r6.json"
APPROXIMATE = ("toxicchat", "rtp_lx_en_response", "xsafety_en")
# (name, size, predictions); the table is sorted by GuardRate (all). Guardrail models only: the untuned d3-mini base
# (runs/d3) is not offered as a guardrail and is left out of the card (its numbers are in experiments.md).
SYSTEMS = [
    ("**AccuKnox Guard 1.1**", "4.5B", "runs/d3-soup59/guardrate_en.jsonl"),
    ("YuFeng-XGuard-Reason-8B", "8B", "runs/xguard/guardrate_en.jsonl"),
    ("YuFeng-XGuard-Reason-0.6B", "0.6B", "runs/xguard-0.6b/guardrate_en.jsonl"),
    ("Qwen3Guard-Gen-8B", "8B", "runs/qwen3guard-8b/guardrate_en.jsonl"),
    ("Qwen3Guard-Gen-4B", "4B", "runs/qwen3guard-4b/guardrate_en.jsonl"),
    ("Qwen3Guard-Gen-0.6B", "0.6B", "runs/qwen3guard-0.6b/guardrate_en.jsonl"),
    ("HaloGuard1-Gen-4B", "4B", "runs/haloguard-4b/guardrate_en.jsonl"),
    ("HaloGuard1-Gen-0.8B", "0.8B", "runs/haloguard-0.8b/guardrate_en.jsonl"),
    ("Sing-Guard-2b", "2B", "runs/sing-guard-2b/guardrate_en.jsonl"),
    ("Shieldstral-1.0-3B", "3B", "runs/shieldstral/guardrate_en.jsonl"),
    ("TypeSafe Jev (API)", "-", "runs/jev/guardrate_en.jsonl"),
    ("AWS Bedrock Guardrails (API)", "-", "runs/bedrock/guardrate_en.jsonl"),
]


def auc(pos, neg):
    """ROC AUC by rank sum, ties averaged."""
    vals = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    rank_sum, i = 0.0, 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j][0] == vals[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(y for _, y in vals[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def p50_latency(path, preds):
    same_gpu = os.path.join(os.path.dirname(path), "latency_a5000.jsonl")
    if os.path.exists(same_gpu):
        return statistics.median(json.loads(l)["latency_ms"] for l in open(same_gpu))
    if "/jev/" in path or "/bedrock/" in path:
        return statistics.median(r["latency_ms"] for r in preds.values())
    raise SystemExit(f"no same-GPU latency for {path}: run scripts/latency_a5000.sh")


def main():
    items = {it["id"]: it for it in map(json.loads, open("data/guardrate_en.jsonl"))}
    groups = {it["dataset"]: it["group"] for it in items.values()}
    seen = set(map(int, json.load(open(SEEN))["flagged"]))
    unseen = {i: it for i, it in items.items() if i not in seen}
    core = {i: it for i, it in unseen.items() if it["dataset"] not in APPROXIMATE}
    rows = []
    for name, size, path in SYSTEMS:
        if not os.path.exists(path) or sum(1 for _ in open(path)) < len(items):
            continue
        preds = load_preds(path, items)
        ranked = {i: r.get("p_graded", r["p_unsafe"]) for i, r in preds.items()}
        ids = list(unseen)
        a = auc([ranked[i] for i in ids if items[i]["unsafe"]], [ranked[i] for i in ids if not items[i]["unsafe"]])
        rows.append((name, size, score_items(unseen, preds, groups, 0.5)[0], score_items(core, preds, groups, 0.5)[0],
                     a, p50_latency(path, preds)))
    print(f"| Model | Size | GuardRate (all) | GuardRate (core) | AUC | p50 latency |\n|---|---|---|---|---|---|")
    for name, size, al, co, a, ms in sorted(rows, key=lambda r: -r[2]):  # best GuardRate (all) first
        print(f"| {name} | {size} | {al:.3f} | {co:.3f} | {a:.3f} | {ms:.0f} ms |")


if __name__ == "__main__":
    main()
