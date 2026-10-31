"""Turn verified training rows into gliner2 classification records.

    python to_gliner.py --input data/train_verified.jsonl --output data/gliner_train.jsonl

Each prompt becomes two single-label tasks:
  block: block | allow   from the dataset's own label (gold_block), which the GLM judges verified
  risk:  one of the 24 categories, from the judge that verified the prompt; always no_risk when allowed
A blocked prompt whose judge chose no_risk is contradictory and is dropped (counted in the log).
Output is gliner2's native format, and every record is checked with gliner2's own validator.
"""

import argparse
import json
import logging
from collections import Counter

from jev_caller import QUESTIONS, RISK_CATEGORIES

log = logging.getLogger("to-gliner")

BLOCK_LABELS = {
    "block": "Harmful request, jailbreak or prompt injection; must be blocked",
    "allow": "Benign; safe to pass to the agent, even if it uses alarming words",
}
BLOCK_PROMPT = QUESTIONS["block"]["instructions"]
RISK_PROMPT = QUESTIONS["risk"]["instructions"]
TASKS = {"block": {"labels": BLOCK_LABELS, "prompt": BLOCK_PROMPT},
         "risk": {"labels": RISK_CATEGORIES, "prompt": RISK_PROMPT}}  # what inference sends


def to_record(text: str, block: bool, risk: str) -> dict:
    return {"input": text, "output": {"classifications": [
        {"task": "block", "labels": list(BLOCK_LABELS), "true_label": ["block" if block else "allow"],
         "prompt": BLOCK_PROMPT, "label_descriptions": BLOCK_LABELS},
        {"task": "risk", "labels": list(RISK_CATEGORIES), "true_label": [risk],
         "prompt": RISK_PROMPT, "label_descriptions": RISK_CATEGORIES},
    ]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="verified rows from verify_glm.py")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from gliner2.training.data import InputExample

    records, dropped, seen = [], Counter(), set()
    for line in open(args.input):
        row = json.loads(line)
        text, block = row["text"].strip(), row["gold_block"]
        risk = row["judge2_risk"] if row["verified_by"] == "judge2" else row["glm_risk"]
        if not block:
            risk = "no_risk"
        elif risk == "no_risk":
            dropped["blocked but judged no_risk"] += 1
            continue
        if text in seen:
            dropped["duplicate"] += 1
            continue
        rec = to_record(text, block, risk)
        errors = InputExample.from_dict(rec).validate()
        if errors:
            raise SystemExit(f"invalid record for {text[:80]!r}: {errors}")
        seen.add(text)
        records.append(rec)

    with open(args.output, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    block = Counter(r["output"]["classifications"][0]["true_label"][0] for r in records)
    risk = Counter(r["output"]["classifications"][1]["true_label"][0] for r in records)
    log.info(f"wrote {len(records)} records, dropped {dict(dropped)} -> {args.output}")
    log.info(f"block: {dict(block)}")
    log.info(f"risk: {dict(risk.most_common())}")


if __name__ == "__main__":
    main()
