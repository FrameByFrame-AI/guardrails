"""Training data for run 8: run 7's rows, each asked the one question its source labels (attack or unsafe content).

    python build_v8.py --train data/v7_train.jsonl --val data/v7_val.jsonl \
        --out-train data/v8_train.jsonl --out-val data/v8_val.jsonl

Runs 5-7 asked one block question (harmful request, jailbreak or injection), but the sources label two different
things: injection/jailbreak sets mark manipulation whatever the topic, content-safety sets mark harmful content
whatever the wrapper. Here each source answers only its own question (jev_caller.ATTACK_QUESTIONS or
CONTENT_QUESTIONS), with its own labels. Response rows keep the response question. The attack question also gets
--attack-negatives rows sampled from the training split's ordinary benign prompts and plain harmful requests
(half each), labelled "not an attack": true by definition (they manipulate nothing), and without them it would
never see a harmful topic that is not an attack. Same texts as run 7, so no new decontamination is needed;
validation is run 7's held-out sources, each on its own question.
"""

import argparse
import json
import random
from collections import Counter

ATTACK = {"gandalf", "jailbreak_classification", "jailbreakv_28k", "llmail_inject", "aeslc_emails",
          "hard_benign_emails", "prompt_shield", "safeguard_injection", "deepset_injection"}
CONTENT = {"aegis", "wildguard", "wildguard_adv_benign", "wildguard_adv_benign_v7", "wildguard_benign", "or_bench",
           "harmfulqa", "salad_base", "categorical_harmfulqa", "coconot"}
BENIGN_POOL = ("or_bench", "wildguard_benign")
HARMFUL_POOL = ("salad_base", "harmfulqa")


def assign(rows):
    out, unknown = [], Counter()
    for r in rows:
        if r["mode"] == "response":
            out.append(r)
        elif r["source"] in ATTACK:
            out.append({**r, "question": "attack"})
        elif r["source"] in CONTENT:
            out.append({**r, "question": "unsafe"})
        else:
            unknown[r["source"]] += 1
    if unknown:
        raise SystemExit(f"sources with no question assigned: {dict(unknown)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", default="data/v7_train.jsonl")
    ap.add_argument("--val", default="data/v7_val.jsonl")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--attack-negatives", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    src = [json.loads(l) for l in open(args.train)]
    train = assign(src)
    for pool in (BENIGN_POOL, HARMFUL_POOL):
        cands = [r for r in src if r["source"] in pool]
        for r in rng.sample(cands, min(args.attack_negatives // 2, len(cands))):
            train.append({**r, "question": "attack", "gold_block": False, "risk": None,
                          "source": r["source"] + ":attack"})
    val = assign(json.loads(l) for l in open(args.val))

    for path, rows in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = Counter((r.get("question", "response"), r["gold_block"]) for r in rows)
        print(f"{len(rows)} rows -> {path}: " + ", ".join(f"{q} {'T' if b else 'F'} {n}" for (q, b), n in sorted(counts.items())))


if __name__ == "__main__":
    main()
