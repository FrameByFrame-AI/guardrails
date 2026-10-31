"""Injection benchmark v2: held-out sources, one family per threat, attacks and benign text.

    python build_injection_v2.py --out data/injection_v2.jsonl
    python guardrate.py audit --input data/injection_v2.jsonl --train <every training file> --out data/injection_v2_seen.json
    python build_injection_v2.py --out data/injection_v2.jsonl --drop-seen data/injection_v2_seen.json

v1 (build_injection_bench.py) overrates our models: run 9 trained on BIPIA's train attacks and was scored on BIPIA's
test attacks, and v1 covers only instructions planted in documents. v2 uses only sources no AccuKnox run trained on,
and the audit step drops any item that matches training text anyway (exact or 50% of 8-word shingles).

Families (dataset ids in brackets; attack-only sets are scored on recall, benign-only on 1 - false-block rate):
- Direct: injections and hijacks typed by the user. CyberSecEval direct [cse_direct], Tensor Trust hijacking and
  prompt extraction attacks [tensortrust_hijack, tensortrust_extract], SPML chatbot attacks of degree >= 3 with its
  benign prompts (degree 0) [spml]. SPML degrees 1-2 are left out: they break a system prompt the guard never sees.
- Indirect: instructions inside documents and tool data. CyberSecEval indirect [cse_indirect], PromptShield test
  attacks and benign prompts [promptshield], texts over 6,000 characters left out.
- Multilingual: CyberSecEval's machine-translated attacks, 16 languages [cse_multilingual], and NotInject's
  multilingual benign prompts [notinject_multilingual] (v1 left them out).
- Over-defence: NotInject's other benign prompts with trigger words [notinject] (data/injection_src, as in v1).
Sources: CyberSecEval (meta-llama/PurpleLlama CybersecurityBenchmarks, MIT; data/injection_v2_src/cyberseceval),
Tensor Trust (qxcv/tensor-trust; no licence stated, used for evaluation only), SPML
(reshabhs/SPML_Chatbot_Prompt_Injection, MIT), PromptShield (hendzh/PromptShield, Apache-2.0), NotInject.
Row format as v1: {id, mode, response, dataset, group, prompt, unsafe}; `group` is the family.
"""

import argparse
import json
import random

SRC = "data/injection_v2_src"
CAP = {"tensortrust_hijack": 400, "tensortrust_extract": 400, "spml": 500, "promptshield": 750, "cse_multilingual": 500}
MAX_CHARS = 6000


def cyberseceval(rng):
    rows = []
    for x in json.load(open(f"{SRC}/cyberseceval/prompt_injection.json")):
        rows.append(("cse_" + x["injection_type"], "Direct" if x["injection_type"] == "direct" else "Indirect",
                     x["user_input"], True))
    ml = [x for x in json.load(open(f"{SRC}/cyberseceval/prompt_injection_multilingual_machine_translated.json"))
          if x["speaking_language"] != "English"]
    rng.shuffle(ml)
    rows += [("cse_multilingual", "Multilingual", x["user_input"], True) for x in ml[:CAP["cse_multilingual"]]]
    return rows


def tensortrust(rng):
    from huggingface_hub import hf_hub_download

    rows = []
    for ds, f in (("tensortrust_hijack", "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"),
                  ("tensortrust_extract", "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl")):
        attacks = sorted({a for l in open(hf_hub_download("qxcv/tensor-trust", f, repo_type="dataset"))
                          if (a := json.loads(l)["attack"])})
        rng.shuffle(attacks)
        rows += [(ds, "Direct", a, True) for a in attacks[:CAP[ds]]]
    return rows


def spml(rng):
    from datasets import load_dataset

    d = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", split="train")
    attacks = sorted({r["User Prompt"] for r in d if r["User Prompt"] and int(r["Degree"]) >= 3})
    benign = sorted({r["User Prompt"] for r in d if r["User Prompt"] and int(r["Degree"]) == 0} - set(attacks))
    rng.shuffle(attacks)
    rng.shuffle(benign)
    return [("spml", "Direct", t, True) for t in attacks[:CAP["spml"]]] + \
           [("spml", "Direct", t, False) for t in benign[:CAP["spml"]]]


def promptshield(rng):
    from datasets import load_dataset

    d = [r for r in load_dataset("hendzh/PromptShield", split="test") if r["prompt"] and len(r["prompt"]) <= MAX_CHARS]
    out = []
    for label in (1, 0):
        texts = sorted({r["prompt"] for r in d if int(r["label"]) == label})
        rng.shuffle(texts)
        out += [("promptshield", "Indirect", t, bool(label)) for t in texts[:CAP["promptshield"]]]
    return out


def notinject():
    import pandas as pd

    rows = []
    for split in ("one", "two", "three"):
        for r in pd.read_parquet(f"data/injection_src/notinject/NotInject_{split}.parquet").to_dict("records"):
            if r["category"] == "Multilingual":  # benign text for the multilingual family (v1 left it out)
                rows.append(("notinject_multilingual", "Multilingual", r["prompt"], False))
            else:
                rows.append(("notinject", "Over-defence", r["prompt"], False))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--drop-seen", help="guardrate.py audit output: drop the flagged items and renumber")
    args = ap.parse_args()
    if args.drop_seen:
        flagged = set(map(int, json.load(open(args.drop_seen))["flagged"]))
        items = [json.loads(l) for l in open(args.out)]
        kept = [it for it in items if it["id"] not in flagged]
        for i, it in enumerate(kept):
            it["id"] = i
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in kept)
        print(f"dropped {len(items) - len(kept)} items seen in training; {len(kept)} left")
        return
    rng = random.Random(args.seed)
    rows = cyberseceval(rng) + tensortrust(rng) + spml(rng) + promptshield(rng) + notinject()
    seen, items = set(), []
    for ds, group, text, unsafe in rows:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            items.append({"id": len(items), "mode": "request", "response": None, "dataset": ds, "group": group,
                          "prompt": text, "unsafe": unsafe})
    with open(args.out, "w") as fh:
        fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
