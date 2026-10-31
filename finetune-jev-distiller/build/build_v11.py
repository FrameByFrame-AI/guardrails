"""Training data for run 11: run 9's data plus conversations, responses and adversarial prompts.

    PYTHONPATH=runs/gblibs GUARDBENCH_DEV_PATH=runs/guardbench_data python build_v11.py \
        --out-train data/v11_train.jsonl --out-val data/v11_val.jsonl

GuardBench (experiments.md, 2026-10-18) showed the gaps: no training row was ever a multi-turn conversation (1.1
wrongly blocks 34% of safe Bot-Adversarial Dialogue user turns given the conversation, 3% given the turn alone),
1.1's parents never trained on a response, and run 10 lost recall on jailbreak-style sets (CSRT, S-Eval attack).
Runs 6-7 showed that added data mostly moves the decision boundary, so every addition is label-balanced within its
kind (conversation, response, adversarial prompt) and carries no risk category (no risk loss).

Added to data/v9_{train,val}.jsonl (labels from the sources, AccuKnox's policy: abuse, hate and harm unsafe; mild
rudeness and grey small talk safe; build_guardbench.graded):
- conversations, train / valid splits only (GuardBench scores their test splits): ProsocialDialog "needs
  intervention" vs "casual", ConvAbuse strongly abusive vs not abusive, Bot-Adversarial Dialogue "ok" turns (safe);
  validation takes a fifth of the training caps from the valid splits; formatted as the evals feed them (build_guardbench.to_item: "User: / Assistant:" transcript, the last assistant
  turn as the response);
- responses: Aegis 2.0 train (response_label, CC-BY-4.0) and WildGuardMix train (response_harm_label, ODC-BY);
- adversarial prompts: WildGuardMix train adversarial harmful vs adversarial benign (ODC-BY).
New rows whose text (prompt or response) matches an evaluation item (exact after normalising, or half or more of
its 8-word shingles in evaluation text; guardrate.py audit's rule) are dropped. Evaluation files: GuardRate EN and
RU, injection v1 and v2, GuardBench, the policy conversation set.
Not added (gated, access needs a request by the user): allenai/wildjailbreak, hackaprompt/hackaprompt-dataset.
"""

import argparse
import json
import random
from collections import Counter

from build_guardbench import graded, to_item
from guardrate import _hf, _norm, _shingles

EVALS = ["data/guardrate_en.jsonl", "data/guardrate_ru.jsonl", "data/injection_bench.jsonl",
         "data/injection_v2.jsonl", "data/guardbench.jsonl", "data/conv_policy_test.jsonl"]
CAP = {"prosocial_dialog": 1500, "convabuse": 600, "bot_adversarial_dialogue": 1000, "aegis_responses": 1000,
       "wildguard_responses": 1000, "wildguard_adversarial": 1500}  # per label (BAD: safe only)
VAL_SHARE = 0.1  # sources without a validation split


def row(source, text, block, response=None):
    return {"mode": "request" if response is None else "response", "response": response, "question": "block",
            "text": text, "gold_block": bool(block), "risk": None, "source": source}


def balanced(rows, cap, rng):
    out = []
    for label in (True, False):
        v = [r for r in rows if r["gold_block"] == label]
        rng.shuffle(v)
        out += v[:cap]
    return out


def conversations(split, rng, scale=1.0):
    by = {}
    for ds, conv, unsafe in graded(split):
        prompt, response = to_item(conv)
        if prompt.strip():
            by.setdefault(ds, []).append(row(ds, prompt, unsafe, response))
    return [r for ds, v in sorted(by.items()) for r in balanced(v, max(1, int(CAP[ds] * scale)), rng)]


def responses(rng):
    unsafe = {"unsafe": True, "safe": False, "harmful": True, "unharmful": False}
    ag = [row("aegis_responses", r["prompt"], unsafe[r["response_label"]], r["response"])
          for r in _hf("nvidia/aegis-ai-content-safety-dataset-2.0", split="train")
          if r["response_label"] in unsafe and r["prompt"] and r["response"]]
    wg_all = list(_hf("allenai/wildguardmix", "wildguardtrain", split="train"))
    wg = [row("wildguard_responses", r["prompt"], unsafe[r["response_harm_label"]], r["response"])
          for r in wg_all if r["response_harm_label"] in unsafe and r["prompt"] and r["response"]]
    adv = [row("wildguard_adversarial", r["prompt"], r["prompt_harm_label"] == "harmful")
           for r in wg_all if r["adversarial"] and r["prompt_harm_label"] in ("harmful", "unharmful") and r["prompt"]]
    return balanced(ag, CAP["aegis_responses"], rng) + balanced(wg, CAP["wildguard_responses"], rng) + \
        balanced(adv, CAP["wildguard_adversarial"], rng)


def eval_index():
    exact, shingles = set(), set()
    for path in EVALS:
        for it in map(json.loads, open(path)):
            for text in (it["prompt"], it.get("response")):
                words = _norm(text or "").split()
                if words:
                    exact.add(" ".join(words))
                    shingles |= _shingles(words)
    return exact, shingles


def seen(r, exact, shingles):
    for text in (r["text"], r["response"]):
        words = _norm(text or "").split()
        sh = _shingles(words)
        if words and (" ".join(words) in exact or (sh and len(sh & shingles) / len(sh) >= 0.5)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-train", default="data/v9_train.jsonl")
    ap.add_argument("--base-val", default="data/v9_val.jsonl")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    new_train = conversations("train", rng)
    new_val = conversations("val", rng, scale=VAL_SHARE * 2)  # validation: a fifth of the training caps
    other = responses(rng)
    rng.shuffle(other)
    k = int(len(other) * VAL_SHARE)
    new_val += other[:k]
    new_train += other[k:]
    exact, shingles = eval_index()
    dropped = Counter()
    for name, rows in (("train", new_train), ("val", new_val)):
        kept = [r for r in rows if not seen(r, exact, shingles)]
        for r in rows:
            if r not in kept:
                dropped[(name, r["source"])] += 1
        rows[:] = kept
    base_train = [json.loads(l) for l in open(args.base_train)]
    base_val = [json.loads(l) for l in open(args.base_val)]
    train, val = base_train + new_train, base_val + new_val
    rng.shuffle(train)
    for path, rows in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"train {len(train)} (run 9 {len(base_train)} + {len(new_train)}), val {len(val)} "
          f"(run 9 {len(base_val)} + {len(new_val)})")
    for name, rows in (("train", new_train), ("val", new_val)):
        c = Counter((r["source"], r["mode"], r["gold_block"]) for r in rows)
        for (src, mode, b), n in sorted(c.items()):
            print(f"  {name:5s} {src:26s} {mode:8s} {'unsafe' if b else 'safe':6s} {n}")
    print("dropped for evaluation overlap:", dict(dropped))


if __name__ == "__main__":
    main()
