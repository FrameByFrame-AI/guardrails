"""Training data for run 12: run 11's data plus HackAPrompt injections (with clean twins) and long benign
instruction prompts.

    python build_v12.py --out-train data/v12_train.jsonl --out-val data/v12_val.jsonl

Run 10 lost injection recall against its base (injection v2 0.782 -> 0.760; InjecAgent base 0.93 -> 0.64) and run
11 adds no injection rows. One change against run 11: successful attacks from HackAPrompt
(hackaprompt/hackaprompt-dataset, MIT; 600k submissions to a public prompt-injection competition, 78k marked
correct), deduplicated (normalised text, then a greedy 8-word-shingle filter: a new attack is skipped when half or
more of its shingles were already taken), sampled evenly over the 11 levels. Each attack appears in one of two
forms, alternating: the attacker's input alone (direct), or the level's full application prompt with the attack
inside (instructions inside an app's template, closer to indirect injection). Each attack has a clean twin: the
same application prompt (or, for the direct form, a plain request) with the attack replaced by a harmless user
input from databricks-dolly-15k (CC-BY-SA-3.0; instructions only), labelled safe. Twins keep the new rows
label-balanced, so the model learns the instruction, not the template.
Second addition (the user asked for fewer runs, so both go into one): run 11 blocks more long, instruction-heavy
safe prompts than run 10 (ToxicChat safe prompts of 300-1,000 characters: 7.4% -> 11.8% blocked: system prompts,
agent tool instructions, persona set-ups), most likely from WildGuardMix's adversarial prompts (persona and rule
scaffolding on both labels). Benign prompts with that structure, at least --min-chars long, labelled safe:
awesome-chatgpt-prompts persona prompts (fka/awesome-chatgpt-prompts, CC0; jailbreak-flavoured ones left out),
function-calling system prompts with the user's first message (glaiveai/glaive-function-calling-v2, Apache-2.0),
Dolly-15k instructions with their reference context (closed QA, extraction, summarisation; CC-BY-SA-3.0).
Rows overlapping an evaluation file are dropped (build_v11's rule); validation takes 10% of attacks with their
twins, and 10% of the instruction prompts.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from build_v11 import eval_index, row, seen
from guardrate import _hf, _norm, _shingles

PER_LEVEL = 200
VAL_SHARE = 0.1
PER_SOURCE = 700  # benign instruction prompts per source
JAILBREAKY = ("jailbreak", "dan ", "no restrictions", "unfiltered", "ignore", "uncensored", "without any limit")


def instruction_prompts(rng, min_chars):
    pools = {"persona_prompts": [r["prompt"] for r in _hf("fka/awesome-chatgpt-prompts", split="train")
                                 if not any(j in r["prompt"].lower() for j in JAILBREAKY)],
             "tool_system_prompts": [], "context_instructions": []}
    for r in _hf("glaiveai/glaive-function-calling-v2", split="train"):
        user = r["chat"].split("ASSISTANT:")[0].replace("USER:", "").strip()
        if r["system"] and user:
            pools["tool_system_prompts"].append(f"{r['system'].strip()}\n\n{user}")
    for r in _hf("databricks/databricks-dolly-15k", split="train"):
        if r["context"] and r["category"] in ("closed_qa", "information_extraction", "summarization"):
            pools["context_instructions"].append(f"{r['instruction'].strip()}\n\n{r['context'].strip()}")
    out = []
    for src, texts in pools.items():
        texts = sorted({t.strip() for t in texts if min_chars <= len(t.strip()) <= 6000})
        rng.shuffle(texts)
        out += [row(src, t, False) for t in texts[:PER_SOURCE]]
    return out


def attacks(rng):
    by_level = defaultdict(list)
    for r in _hf("hackaprompt/hackaprompt-dataset", split="train"):
        u = (r["user_input"] or "").strip()
        if r["correct"] and u and r["prompt"] and u in r["prompt"]:
            by_level[r["level"]].append((u, r["prompt"]))
    out = []
    for level in sorted(by_level):
        v = by_level[level]
        rng.shuffle(v)
        taken, exact, picked = set(), set(), []
        for u, prompt in v:
            words = _norm(u).split()
            key, sh = " ".join(words), _shingles(words)
            if key in exact or (sh and len(sh & taken) >= 0.5 * len(sh)):
                continue
            exact.add(key)
            taken |= sh
            picked.append((level, u, prompt))
            if len(picked) == PER_LEVEL:
                break
        out += picked
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-train", default="data/v11_train.jsonl")
    ap.add_argument("--base-val", default="data/v11_val.jsonl")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--min-chars", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    benign = [r["instruction"].strip() for r in _hf("databricks/databricks-dolly-15k", split="train")
              if r["instruction"] and len(r["instruction"]) < 400]
    rng.shuffle(benign)
    pairs = []
    for i, (level, u, prompt) in enumerate(attacks(rng)):
        b = benign[i]
        if i % 2 == 0:
            pairs.append([row("hackaprompt_direct", u, True), row("hackaprompt_direct_twin", b, False)])
        else:
            pairs.append([row("hackaprompt_app", prompt, True),
                          row("hackaprompt_app_twin", prompt.replace(u, b), False)])
    rng.shuffle(pairs)
    k = int(len(pairs) * VAL_SHARE)
    exact, shingles = eval_index()
    dropped = Counter()

    def keep(ps):
        out = []
        for p in ps:
            if any(seen(r, exact, shingles) for r in p):
                dropped[p[0]["source"]] += 1
                continue
            out += p
        return out
    new_val, new_train = keep(pairs[:k]), keep(pairs[k:])
    instr = [r for r in instruction_prompts(rng, args.min_chars) if not seen(r, exact, shingles)]
    rng.shuffle(instr)
    k2 = int(len(instr) * VAL_SHARE)
    new_val += instr[:k2]
    new_train += instr[k2:]
    base_train = [json.loads(l) for l in open(args.base_train)]
    base_val = [json.loads(l) for l in open(args.base_val)]
    train, val = base_train + new_train, base_val + new_val
    rng.shuffle(train)
    for path, rows in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"train {len(train)} (run 11 {len(base_train)} + {len(new_train)}), val {len(val)} "
          f"(run 11 {len(base_val)} + {len(new_val)})")
    for name, rows in (("train", new_train), ("val", new_val)):
        for (src, b), n in sorted(Counter((r["source"], r["gold_block"]) for r in rows).items()):
            print(f"  {name:5s} {src:26s} {'unsafe' if b else 'safe':6s} {n}")
    print("pairs dropped for evaluation overlap:", dict(dropped))


if __name__ == "__main__":
    main()
