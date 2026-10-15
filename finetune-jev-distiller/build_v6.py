"""Training data for run 6: run 5's data minus two sources, plus plain harmful requests and response pairs,
decontaminated against every eval item.

    python build_v6.py --out data/train_v6.jsonl

- Kept: data/train_verified.jsonl (run 5's verified prompts) without or_bench_toxic (it is a GuardRate eval set)
  and toxic_chat (CC-BY-NC-4.0).
- Added, plain harmful requests (gold by construction, Apache-2.0): OpenSafetyLab/Salad-Data base_set (sampled,
  stratified by its top-level category), declare-lab/CategoricalHarmfulQA English, declare-lab/HarmfulQA.
- Added, response moderation (train splits only; their test splits are GuardRate sets): allenai/wildguardmix
  wildguardtrain (response_harm_label, ODC-BY) and Aegis 2.0 train (response_label, human, CC-BY-4.0), each
  balanced safe/unsafe.
- Decontamination: an item is dropped when its prompt or response matches an eval item exactly after normalising
  case, punctuation and whitespace, or when half or more of its 8-word shingles occur in eval text (catches
  near-duplicates and shared templates; guardrate.py audit uses the same rule in the other direction). Eval text:
  data/guardrate_en.jsonl (prompts and responses) and data/benchmark_candidates.jsonl.
Output rows: mode (request | response), text, response (or null), gold_block, risk (or null when the source has
no category label), source, group. Labels are the datasets' own; nothing is sent to an LLM.
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict

from guardrate import _norm, _shingles

log = logging.getLogger("build-v6")
DROP_SOURCES = {"or_bench_toxic": "GuardRate eval set", "toxic_chat": "CC-BY-NC-4.0"}


def eval_index(paths_fields):
    exact, shingles = set(), set()
    for path, fields in paths_fields:
        for line in open(path):
            r = json.loads(line)
            for f in fields:
                if r.get(f):
                    words = _norm(r[f]).split()
                    exact.add(" ".join(words))
                    shingles |= _shingles(words)
    return exact, shingles


def seen(text, exact, shingles, overlap):
    if not text:
        return False
    words = _norm(text).split()
    if " ".join(words) in exact:
        return True
    sh = _shingles(words)
    return bool(sh) and len(sh & shingles) / len(sh) >= overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--verified", default="data/train_verified.jsonl")
    ap.add_argument("--salad", type=int, default=2000)
    ap.add_argument("--harmfulqa", type=int, default=800)
    ap.add_argument("--responses-per-source", type=int, default=2000, help="half safe, half unsafe")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import datasets

    rng = random.Random(args.seed)
    rows = []

    # 1. run 5's verified prompts, minus the dropped sources
    dropped_src = Counter()
    for r in map(json.loads, open(args.verified)):
        if r["source"] in DROP_SOURCES:
            dropped_src[r["source"]] += 1
            continue
        risk = r["judge2_risk"] if r["verified_by"] == "judge2" else r["glm_risk"]
        rows.append({"mode": "request", "text": r["text"].strip(), "response": None, "gold_block": r["gold_block"],
                     "risk": "no_risk" if not r["gold_block"] else (None if risk == "no_risk" else risk),
                     "source": r["source"], "group": r["group"]})
    log.info(f"run 5 prompts kept {len(rows)}; removed {dict(dropped_src)} ({DROP_SOURCES})")

    def add_requests(source, texts):
        n = 0
        for t in texts:
            if t and t.strip():
                rows.append({"mode": "request", "text": t.strip(), "response": None, "gold_block": True,
                             "risk": None, "source": source, "group": "harmful_requests"})
                n += 1
        log.info(f"{source:24s} {n}")

    # 2. plain harmful requests
    salad = list(datasets.load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train"))
    by_cat = defaultdict(list)
    for r in salad:
        by_cat[r["1-category"]].append(r["question"])
    per = args.salad // len(by_cat)
    add_requests("salad_base", [q for qs in by_cat.values() for q in rng.sample(qs, min(per, len(qs)))])
    add_requests("categorical_harmfulqa",
                 [r["Question"] if "Question" in r else r["question"]
                  for r in datasets.load_dataset("declare-lab/CategoricalHarmfulQA", split="en")])
    hq = [r["question"] for r in datasets.load_dataset("declare-lab/HarmfulQA", split="train")]
    add_requests("harmfulqa", rng.sample(hq, min(args.harmfulqa, len(hq))))

    # 3. response pairs, balanced safe/unsafe per source
    def add_responses(source, group, pairs):
        half = args.responses_per_source // 2
        unsafe = [p for p in pairs if p[2]]
        safe = [p for p in pairs if not p[2]]
        picked = rng.sample(unsafe, min(half, len(unsafe))) + rng.sample(safe, min(half, len(safe)))
        for prompt, response, bad in picked:
            rows.append({"mode": "response", "text": prompt.strip(), "response": response.strip(), "gold_block": bad,
                         "risk": None, "source": source, "group": group})
        log.info(f"{source:24s} {len(picked)} ({sum(p[2] for p in picked)} unsafe)")

    wg = datasets.load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    add_responses("wildguard_responses", "responses",
                  [(r["prompt"], r["response"], r["response_harm_label"] == "harmful") for r in wg
                   if r["prompt"] and r["response"] and r["response_harm_label"] in ("harmful", "unharmful")])
    ag = datasets.load_dataset("nvidia/aegis-ai-content-safety-dataset-2.0", split="train")
    add_responses("aegis_responses", "responses",
                  [(r["prompt"], r["response"], r["response_label"] == "unsafe") for r in ag
                   if r["prompt"] and r["response"] and r["response_label"] in ("safe", "unsafe")])

    # 4. decontaminate against every eval item, and drop duplicates
    exact, shingles = eval_index([("data/guardrate_en.jsonl", ("prompt", "response")),
                                  ("data/benchmark_candidates.jsonl", ("text",))])
    out, removed, dup, keys = [], Counter(), Counter(), set()
    for r in rows:
        if seen(r["text"], exact, shingles, args.overlap) or seen(r["response"], exact, shingles, args.overlap):
            removed[r["source"]] += 1
            continue
        k = (r["mode"], _norm(r["text"]), _norm(r["response"] or ""))
        if k in keys:
            dup[r["source"]] += 1
            continue
        keys.add(k)
        out.append(r)
    with open(args.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = Counter(r["source"] for r in rows)
    kept = Counter(r["source"] for r in out)
    log.info(f"{'source':26s} {'built':>6} {'eval overlap':>12} {'duplicate':>9} {'kept':>6}")
    for s in sorted(total, key=lambda s: -total[s]):
        log.info(f"{s:26s} {total[s]:6d} {removed[s]:12d} {dup[s]:9d} {kept[s]:6d}")
    by_mode = Counter((r["mode"], r["gold_block"]) for r in out)
    log.info(f"wrote {len(out)} rows -> {args.out}: {dict(by_mode)}; removed {sum(removed.values())} for eval overlap")


if __name__ == "__main__":
    main()
