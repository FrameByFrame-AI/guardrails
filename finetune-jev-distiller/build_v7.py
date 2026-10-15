"""Training data for run 7: run 6's data plus benign WildGuardMix train prompts, to restore the benign-to-harmful
balance run 6 lost (its false blocks doubled), decontaminated the same way.

    python build_v7.py --v6 data/train_v6.jsonl --out data/train_v7.jsonl

Run 5 had 1.7 benign ordinary chat prompts per harmful one, run 6 1.0 (toxic-chat's benign prompts left, plain
harmful requests came in). Added: allenai/wildguardmix wildguardtrain prompts labelled unharmful (ODC-BY, the
dataset's own labels), --vanilla plain and --adversarial adversarial ones, not already in run 6's data. Dropped like
build_v6.py: exact or near-duplicate (half or more of the 8-word shingles) of any GuardRate item or internal
benchmark candidate, and duplicates.
"""

import argparse
import json
import logging
import random
from collections import Counter

from build_v6 import eval_index, seen
from guardrate import _norm

log = logging.getLogger("build-v7")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v6", default="data/train_v6.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vanilla", type=int, default=3000)
    ap.add_argument("--adversarial", type=int, default=1000)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import datasets

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.v6)]
    keys = {(r["mode"], _norm(r["text"]), _norm(r["response"] or "")) for r in rows}
    exact, shingles = eval_index([("data/guardrate_en.jsonl", ("prompt", "response")),
                                  ("data/benchmark_candidates.jsonl", ("text",))])

    wg = datasets.load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    pools = {False: [], True: []}
    for r in wg:
        if r["prompt_harm_label"] == "unharmful" and r["prompt"] and r["prompt"].strip():
            pools[bool(r["adversarial"])].append(r["prompt"].strip())
    stats = Counter()
    for adversarial, want, source in ((False, args.vanilla, "wildguard_benign"),
                                      (True, args.adversarial, "wildguard_adv_benign_v7")):
        pool = pools[adversarial]
        rng.shuffle(pool)
        n = 0
        for text in pool:
            if n >= want:
                break
            k = ("request", _norm(text), "")
            if k in keys:
                stats[source + " duplicate"] += 1
                continue
            if seen(text, exact, shingles, args.overlap):
                stats[source + " eval overlap"] += 1
                continue
            keys.add(k)
            rows.append({"mode": "request", "text": text, "response": None, "gold_block": False, "risk": "no_risk",
                         "source": source, "group": "over_refusal"})
            n += 1
        stats[source + " added"] = n
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"{dict(stats)}")
    log.info(f"wrote {len(rows)} rows -> {args.out}: "
             f"{dict(Counter((r['mode'], r['gold_block']) for r in rows))}")


if __name__ == "__main__":
    main()
