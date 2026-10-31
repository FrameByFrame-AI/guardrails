"""Second opinion on the gold-labelled prompts GLM dropped: label them blind with a stronger model of another family.

    python adjudicate.py --candidates data/benchmark_candidates.jsonl --verified data/benchmark.jsonl \
        --cache data/opus_labels.jsonl --output data/eval_adjudicated.jsonl

Only prompts from datasets kept by EVAL.md are sent (the rest are dropped for their label provenance anyway).
The adjudicator gets verify_glm.py's exact instructions and schema and never sees the dataset label. Outcome per
prompt: "restore" when it agrees with the dataset label (back into the eval), "review" when it sides with GLM
against the label (a person decides). Labels are cached append-only, so a prompt is never paid for twice.
"""

import argparse
import json
import logging
import os
from collections import Counter

from verify_glm import label_all

log = logging.getLogger("adjudicate")
EVAL_DATASETS = {"wildguard", "aegis", "toxic_chat", "openai_moderation", "xs_test", "harmbench", "jbb",
                 "in_the_wild", "chatgpt_jailbreaks", "jailbreak_classification", "deepset_injection",
                 "llmail_inject", "gandalf", "aeslc_emails", "jailbreakv_28k"}  # the keep list in EVAL.md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--glm-cache", default="data/glm_labels.jsonl")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="anthropic/claude-opus-5.5")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    key = os.environ.get("OPENROUTER_API_KEY") or exit("OPENROUTER_API_KEY not set")

    kept = {json.loads(l)["text"].strip() for l in open(args.verified)}
    rows = [r for r in map(json.loads, open(args.candidates))
            if r["source"] in EVAL_DATASETS and r["text"].strip() not in kept]
    glm = {}
    for l in open(args.glm_cache):
        c = json.loads(l)
        if c["model"] == "z-ai/glm-5.3-flash":
            glm[c["text"].strip()] = c["block"]
    log.info(f"{len(rows)} GLM-dropped prompts from eval datasets")

    labels, errors = label_all([r["text"] for r in rows], args.model, args.provider, True, args.cache,
                               args.workers, key, temperature=False)  # Opus 5.5 rejects temperature
    out, outcome, cost = [], Counter(), 0.0
    for r in rows:
        lab = labels.get(r["text"])
        if lab is None:
            continue
        o = "restore" if lab["block"] == r["gold_block"] else "review"
        outcome[(r["source"], o)] += 1
        cost += lab.get("cost") or 0.0
        out.append({**r, "glm_block": glm.get(r["text"].strip()), "adjudicator": args.model,
                    "adjudicator_block": lab["block"], "adjudicator_risk": lab["risk"], "outcome": o})
    with open(args.output, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = Counter(r["outcome"] for r in out)
    log.info(f"{dict(total)} of {len(out)}; {errors} failed calls; cost ${cost:.2f} -> {args.output}")
    for s in sorted({s for s, _ in outcome}):
        log.info(f"  {s:26s} restore {outcome[(s, 'restore')]:3d}  review {outcome[(s, 'review')]:3d}")
    if errors:
        raise SystemExit(f"{errors} calls failed; rerun to retry (cached labels are kept)")


if __name__ == "__main__":
    main()
