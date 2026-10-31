"""GuardBench (AmenRa/guardbench, EMNLP 2024) in guardrate.py's item format, so `guardrate.py run` scores it.

    PYTHONPATH=runs/gblibs GUARDBENCH_DEV_PATH=runs/guardbench_data python build_guardbench.py build --out data/guardbench.jsonl
    python guardrate.py run --model d3 --path <package> --input data/guardbench.jsonl --output runs/<dir>/guardbench.jsonl
    PYTHONPATH=runs/gblibs python build_guardbench.py score --input data/guardbench.jsonl --system name=runs/<dir>/guardbench.jsonl

The library (guardbench==1.0.1, runs/gblibs, installed --no-deps) downloads and formats its datasets; datasets that
fail (gated ones need a form: DecodingTrust, HEx-PHI, HarmEval, TechHazardQA, ToxiGen) are listed and skipped.
Conversations become one item each: a single user turn is a request; a conversation ending in an assistant turn is
a response, the earlier turns joined as the prompt ("User: ..." / "Assistant: ..." lines); a longer conversation
ending in a user turn is a request over the whole transcript. Scored with the library's own evaluate() at 0.5,
as its Report does: F1 per dataset, recall on unsafe-only datasets (F1 is undefined there), plus false-positive
rate; the leaderboard-style average is the mean of those per-dataset numbers.
"""

import argparse
import json


def transcript(turns):
    return "\n".join(f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}" for t in turns)


def to_item(conv):
    if len(conv) == 1:
        return conv[0]["content"], None
    if conv[-1]["role"] == "assistant":
        head = conv[:-1]
        return (head[0]["content"] if len(head) == 1 else transcript(head)), conv[-1]["content"]
    return transcript(conv), None


def cmd_build(args):
    from guardbench.datasets import DATASETS, load_dataset

    items, failed = [], {}
    for alias in DATASETS:
        try:
            ds = load_dataset(alias)
        except Exception as e:  # gated or unavailable
            failed[alias] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        for x in ds:
            prompt, response = to_item(x["conversation"])
            if not prompt.strip():
                continue
            items.append({"id": len(items), "mode": "request" if response is None else "response",
                          "response": response, "dataset": alias, "group": ds.category, "prompt": prompt,
                          "unsafe": bool(x["label"]), "gb_id": x["id"]})
        print(f"{alias}: {len(ds)}", flush=True)
    with open(args.out, "w") as fh:
        fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    print(f"{len(items)} items -> {args.out}; skipped {len(failed)}:")
    for k, v in failed.items():
        print(f"  {k}: {v}")


def graded(split):
    """(dataset, conversation, unsafe) under AccuKnox's policy (abuse, hate and harm unsafe; mild rudeness and grey
    small talk safe), clear cases only. ProsocialDialog: __casual__ safe, __needs_intervention__ unsafe, the three
    caution grades dropped. ConvAbuse: every annotator "not abusive" (1) safe, a majority "strongly" or "very
    strongly" abusive (-2, -3) unsafe, the rest dropped. Bot-Adversarial Dialogue: "ok" turns only, as safe (its
    "notok" covers mild rudeness). DICES 350 / 990 (test only): GuardBench's labels (a majority of raters judge the
    reply unsafe or biased)."""
    from guardbench.datasets import load_dataset
    from guardbench.datasets.convabuse import ConvAbuse
    from guardbench.datasets.prosocial_dialog import ProsocialDialog

    pd_ = ProsocialDialog()
    pd_.convert_label = lambda x: {"__casual__": False, "__needs_intervention__": True}.get(x)
    for x in pd_.format_data(pd_.load_raw_data())[split]:
        if x["label"] is not None:
            yield "prosocial_dialog", x["conversation"], x["label"]

    ca = ConvAbuse()

    def severity(x):
        sev = [next(j for j in range(-3, 2) if x[f"Annotator{i}_is_abuse.{j}"] == "1") for i in range(1, 9)
               if any(x[f"Annotator{i}_is_abuse.{j}"] == "1" for j in range(-3, 2))]
        if all(s == 1 for s in sev):
            return False
        return True if sum(s <= -2 for s in sev) / len(sev) > 0.5 else None
    ca.compute_label = severity
    for x in ca.format_data(ca.load_raw_data())[split]:
        if x["label"] is not None:
            yield "convabuse", x["conversation"], x["label"]

    for x in load_dataset("bot_adversarial_dialogue").load(split):
        if not x["label"]:
            yield "bot_adversarial_dialogue", x["conversation"], False
    if split == "test":
        for alias in ("dices_350", "dices_990"):
            for x in load_dataset(alias):
                yield alias, x["conversation"], bool(x["label"])


def cmd_policy(args):
    items = []
    for ds, conv, unsafe in graded(args.split):
        prompt, response = to_item(conv)
        if prompt.strip():
            items.append({"id": len(items), "mode": "request" if response is None else "response",
                          "response": response, "dataset": ds, "group": "conversations", "prompt": prompt,
                          "unsafe": unsafe})
    with open(args.out, "w") as fh:
        fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    from collections import Counter
    for (ds, u), n in sorted(Counter((it["dataset"], it["unsafe"]) for it in items).items()):
        print(f"{ds:28s} {'unsafe' if u else 'safe':6s} {n}")
    print(f"{len(items)} items -> {args.out}")


def cmd_score(args):
    from guardbench.evaluate import evaluate

    items = [json.loads(l) for l in open(args.input)]
    seen = set(map(int, json.load(open(args.seen))["flagged"])) if args.seen else set()
    datasets = sorted({it["dataset"] for it in items})
    table = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        preds = {r["id"]: r["p_unsafe"] for r in map(json.loads, open(path))}
        table[name] = {}
        for d in datasets:
            rows = [it for it in items if it["dataset"] == d and it["id"] not in seen and it["id"] in preds]
            if rows:
                table[name][d] = evaluate({r["id"]: r["unsafe"] for r in rows}, {r["id"]: preds[r["id"]] for r in rows})
    names = list(table)
    print("| dataset | n | unsafe | " + " | ".join(f"{n} F1 / recall / FPR" for n in names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for d in datasets:
        r0 = table[names[0]].get(d)
        if not r0:
            continue
        cells = []
        for n in names:
            r = table[n][d]
            cells.append(f"{r['f1']:.3f} / {r['recall']:.3f} / {r['fpr']:.3f}" if r["n"] else f"– / {r['recall']:.3f} / –")
        print(f"| {d} | {int(r0['p'] + r0['n'])} | {int(r0['p'])} | " + " | ".join(cells) + " |")
    mixed = [d for d in datasets if table[names[0]].get(d, {}).get("n")]
    only = [d for d in datasets if d in table[names[0]] and not table[names[0]][d]["n"]]
    print()
    for n in names:
        f1 = sum(table[n][d]["f1"] for d in mixed) / len(mixed)
        rec = sum(table[n][d]["recall"] for d in only) / len(only)
        both = (f1 * len(mixed) + rec * len(only)) / (len(mixed) + len(only))
        print(f"{n}: leaderboard-style mean {both:.3f} over {len(mixed) + len(only)} datasets (F1 on {len(mixed)} "
              f"mixed {f1:.3f}, recall on {len(only)} unsafe-only {rec:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    p = sub.add_parser("policy", help="conversations relabelled to AccuKnox's policy (see graded())")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--input", required=True)
    s.add_argument("--system", action="append", required=True, help="name=predictions.jsonl")
    s.add_argument("--seen", help="guardrate.py audit output: leave these items out")
    args = ap.parse_args()
    {"build": cmd_build, "policy": cmd_policy, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
