"""Score guardrail systems against the benchmark's dataset labels, with measured latency and cost.

    python compare_systems.py --bench data/benchmark.jsonl --jev data/jev_labels.jsonl \
        --glm data/glm_labels.jsonl --bedrock data/bedrock_labels.jsonl \
        --system gold-gliner=runs/gold-gliner/benchmark.jsonl --output runs/compare.json

Each system is read from its cache (keyed by prompt text); a system is skipped if its cache is not given.
Metrics per system: accuracy, precision, recall, F1, false-block rate (benign prompts blocked), coverage (prompts
answered), the same per group, latency p50/p95 and cost per 1,000 prompts. On a GLM-verified benchmark the GLM
row is circular (GLM chose which prompts agree with the labels) and is marked as such.
"""

import argparse
import json
from collections import defaultdict


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def load_systems(args) -> dict[str, dict[str, dict]]:
    systems = {}
    if args.jev:
        systems["jev"] = {r["text"].strip(): {"block": r["jev"]["block_prob"] >= args.jev_threshold,
                                              "cost": r["jev"].get("cost"), "latency_ms": r["jev"].get("latency_ms")}
                          for r in read_jsonl(args.jev)}
    if args.glm:
        systems["glm"] = {r["text"]: r for r in read_jsonl(args.glm)
                          if r["model"] == args.glm_model and r.get("provider_slug") == args.glm_provider}
    if args.bedrock:
        systems["bedrock"] = {r["text"]: r for r in read_jsonl(args.bedrock)}
    for spec in args.system:
        name, _, path = spec.partition("=")
        systems[name] = {r["text"]: r for r in read_jsonl(path)}
    return systems


def metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(p and g for p, g in pairs)
    fp = sum(p and not g for p, g in pairs)
    fn = sum(not p and g for p, g in pairs)
    tn = sum(not p and not g for p, g in pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(pairs), "accuracy": round((tp + tn) / len(pairs), 4) if pairs else 0.0,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "false_block_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0}


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, help="rows with text, gold_block, group, source")
    ap.add_argument("--jev")
    ap.add_argument("--jev-threshold", type=float, default=0.5)
    ap.add_argument("--glm")
    ap.add_argument("--glm-model", default="z-ai/glm-5.3-flash")
    ap.add_argument("--glm-provider", default="fireworks")
    ap.add_argument("--bedrock")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=PATH",
                    help="any other system's predictions (rows with text, block, latency_ms, cost), e.g. from gliner_caller.py")
    ap.add_argument("--output")
    ap.add_argument("--cases", action="store_true", help="also print every prompt's decisions")
    args = ap.parse_args()

    bench = read_jsonl(args.bench)
    systems = load_systems(args)
    glm_verified = any("glm_block" in r for r in bench)
    report = {"bench": args.bench, "n": len(bench), "systems": {}}

    for name, cache in systems.items():
        answered = [r for r in bench if r["text"] in cache]
        if not answered:
            continue
        by_group = defaultdict(list)
        for r in answered:
            by_group[r["group"]].append((cache[r["text"]]["block"], r["gold_block"]))
        costs = [cache[r["text"]]["cost"] for r in answered if cache[r["text"]].get("cost") is not None]
        lats = [cache[r["text"]]["latency_ms"] for r in answered if cache[r["text"]].get("latency_ms") is not None]
        report["systems"][name] = {
            "coverage": round(len(answered) / len(bench), 4),
            **metrics([(cache[r["text"]]["block"], r["gold_block"]) for r in answered]),
            "by_group": {g: metrics(v) for g, v in sorted(by_group.items())},
            "latency_ms": {"p50": pct(lats, 0.5), "p95": pct(lats, 0.95), "measured": len(lats)},
            "cost_per_1k_prompts_usd": round(1000 * sum(costs) / len(costs), 4) if costs else None,
            "circular": name == "glm" and glm_verified,
        }

    print(f"{len(bench)} benchmark prompts\n")
    hdr = f"{'system':<14}{'cover':>6}{'acc':>7}{'prec':>7}{'recall':>7}{'f1':>7}{'FBR':>7}{'p50 ms':>8}{'p95 ms':>8}{'$/1k':>9}"
    print(hdr)
    for name, m in report["systems"].items():
        lat = m["latency_ms"]
        cost = f"{m['cost_per_1k_prompts_usd']:.3f}" if m["cost_per_1k_prompts_usd"] is not None else "-"
        print(f"{name + ('*' if m['circular'] else ''):<14}{m['coverage']:>6.2f}{m['accuracy']:>7.3f}{m['precision']:>7.3f}"
              f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['false_block_rate']:>7.3f}"
              f"{lat['p50'] if lat['p50'] is not None else '-':>8}{lat['p95'] if lat['p95'] is not None else '-':>8}{cost:>9}")
    if any(m["circular"] for m in report["systems"].values()):
        print("* GLM verified this benchmark, so its own score here is circular.")
    groups = sorted({g for m in report["systems"].values() for g in m["by_group"]})
    for metric in ("accuracy", "recall", "false_block_rate"):
        print(f"\n{metric + ' by group':<26}{'n':>5}" + "".join(f"{n:>13}" for n in report["systems"]))
        for g in groups:
            n = max(m["by_group"].get(g, {}).get("n", 0) for m in report["systems"].values())
            print(f"{g:<26}{n:>5}" + "".join(f"{m['by_group'][g][metric]:>13.2f}" if g in m["by_group"] else f"{'-':>13}"
                                           for m in report["systems"].values()))
    if args.cases:
        print(f"\n{'case':<32}{'gold':>6}" + "".join(f"{n:>9}" for n in systems))
        for r in bench:
            dec = [("block" if systems[n][r["text"]]["block"] else "allow") if r["text"] in systems[n] else "-"
                   for n in systems]
            print(f"{r.get('id', r['text'][:30]):<32}{'block' if r['gold_block'] else 'allow':>6}"
                  + "".join(f"{d:>9}" for d in dec))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
