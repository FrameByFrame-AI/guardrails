"""int8 on CPU: does dynamic quantisation keep a decision model's scores, and how fast is it?

    python cpu_int8.py run --model gliner --precision fp32 --threads 8
    python cpu_int8.py run --model d3-lite --precision int8 --threads 8
    python cpu_int8.py report

Scores a fixed sample of GuardRate English request items (seeded, unseen by runs 5/6) with the block question
only, one item at a time, and writes runs/cpu/<model>-<precision>.jsonl (resumable). int8 is PyTorch dynamic
quantisation of every nn.Linear (weights int8 with one scale per tensor, activations quantised per call);
int8-pc uses one scale per output channel; int8-sel is per-channel on the bulk only (GLiNER: the encoder, heads
fp32; d3: the language model's MLPs, attention and linear-attention projections fp32); d3 packages are converted from
their fixed BF16 to fp32 first, since this CPU has no fast BF16. `report` compares each int8 run with its fp32 run:
probability drift, decision flips at 0.5, AUC against the gold labels, latency by input length.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

OUT = Path("runs/cpu")
PRECISIONS = ["fp32", "int8", "int8-pc", "int8-sel"]
MODELS = {"gliner": "/models/GLiNER2.5-Decide", "d3-lite": "vllm-sr/d3-lite", "d3-lite-v9": "runs/d3-lite-v9/package"}


def sample(n, seed=7):
    seen_ids = set(map(int, json.load(open("data/guardrate_en_seen_r5r6.json"))["flagged"]))
    rows = [json.loads(l) for l in open("data/guardrate_en.jsonl")]
    rows = [r for r in rows if r["mode"] == "request" and r["id"] not in seen_ids]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load(name, precision):
    if name == "gliner":
        from gliner2 import AutoExtractor
        from to_gliner import TASKS
        model = AutoExtractor.from_pretrained(MODELS[name]).eval()
        core = model
        tasks = {"block": TASKS["block"]}

        def run(text):
            with torch.inference_mode():
                b = model.classify_text(text, tasks, include_confidence=True, max_len=1024)["block"]
            return b["confidence"] if b["label"] == "block" else 1.0 - b["confidence"]
    else:
        from transformers import AutoModel
        from jev_caller import QUESTIONS
        model = AutoModel.from_pretrained(MODELS[name], trust_remote_code=True)
        core = model.runtime.backbone.float()  # the package refuses casts; convert the backbone itself
        qs = {"block": QUESTIONS["block"]}

        def run(text):
            with torch.inference_mode():
                a = model.system_one(state=text, questions=qs)["answers"]["block"]
            if "error" in a:
                raise RuntimeError(f"d3: {a['error']}: {a.get('message')}")
            return a["noul"]
    if precision != "fp32":
        from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig
        qconfig = default_dynamic_qconfig if precision == "int8" else per_channel_dynamic_qconfig
        names = [n for n, m in core.named_modules() if isinstance(m, torch.nn.Linear)]
        if precision == "int8-sel":
            names = [n for n in names if (n.startswith("encoder.") if name == "gliner" else ".mlp." in n and "visual" not in n)]
        spec = torch.nn.Linear if precision != "int8-sel" else None
        torch.ao.quantization.quantize_dynamic(core, {spec: qconfig} if spec else {n: qconfig for n in names},
                                               dtype=torch.qint8, inplace=True)
        print(f"{name}: {len(names)} Linear layers quantised ({precision})", flush=True)
    return run


def cmd_run(args):
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}-{args.precision}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    items = [r for r in sample(args.n) if r["id"] not in done]
    print(f"{args.model} {args.precision}: {len(done)} done, {len(items)} to go", flush=True)
    if not items:
        return
    run = load(args.model, args.precision)
    run(items[0]["prompt"])  # warm-up
    with open(path, "a") as fh:
        for r in items:
            t = time.perf_counter()
            p = run(r["prompt"])
            ms = (time.perf_counter() - t) * 1000
            fh.write(json.dumps({"id": r["id"], "dataset": r["dataset"], "unsafe": r["unsafe"], "chars": len(r["prompt"]),
                                 "p": round(float(p), 5), "ms": round(ms, 1)}) + "\n")
            fh.flush()


def auc(rows):
    pos = [r["p"] for r in rows if r["unsafe"]]
    neg = [r["p"] for r in rows if not r["unsafe"]]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def cmd_report(args):
    def load_rows(m, p):
        f = OUT / f"{m}-{p}.jsonl"
        return {r["id"]: r for r in map(json.loads, open(f))} if f.exists() else {}

    print("| model | precision | n | AUC | accuracy@0.5 | p50 ms (<300 chars) | p50 ms (>=300 chars) | p95 ms |")
    print("|---|---|---|---|---|---|---|---|")
    drift = []
    for m in MODELS:
        runs = {p: load_rows(m, p) for p in PRECISIONS}
        for p, rows in runs.items():
            if not rows:
                continue
            v = list(rows.values())
            short = [r["ms"] for r in v if r["chars"] < 300] or [0]
            long = [r["ms"] for r in v if r["chars"] >= 300] or [0]
            ms = sorted(r["ms"] for r in v)
            acc = sum((r["p"] >= 0.5) == r["unsafe"] for r in v) / len(v)
            print(f"| {m} | {p} | {len(v)} | {auc(v):.3f} | {acc:.3f} | {statistics.median(short):.0f} | "
                  f"{statistics.median(long):.0f} | {ms[int(0.95 * (len(ms) - 1))]:.0f} |")
        for q in PRECISIONS[1:]:
            common = runs["fp32"].keys() & runs[q].keys()
            if common:
                d = [abs(runs["fp32"][i]["p"] - runs[q][i]["p"]) for i in common]
                flips = sum((runs["fp32"][i]["p"] >= 0.5) != (runs[q][i]["p"] >= 0.5) for i in common)
                drift.append(f"{m} {q}: {len(common)} items, mean |dp| {statistics.mean(d):.4f}, max {max(d):.4f}, "
                             f"decision flips {flips}")
    print("\n" + "\n".join(drift))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=list(MODELS), required=True)
    r.add_argument("--precision", choices=PRECISIONS, required=True)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--n", type=int, default=300)
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
