"""Training data for run 9: run 5's data plus documents carrying planted instructions, with their clean twins.

    python build_v9.py --out-train data/v9_train.jsonl --out-val data/v9_val.jsonl

Run 5 misses 41% of the public injection benchmark's BIPIA attacks, mostly planted instructions whose content is
harmless ("translate the following sentence", "write a slogan"); its only document-injection source was LLMail
(exfiltration by email). One change: run 5's rows (data/split_{train,val}.jsonl, converted to build_v6's row
format, sources from train_verified.jsonl) plus
- attacks: BIPIA's *train* attack strings (75 text, 50 code; no string shared with its test split, one category
  name shared, "Language Translation", with differently phrased strings) planted at the start, middle or end of
  a document: text attacks into emails (BIPIA train emails and run 5's own aeslc / hard benign training emails, whose
  clean versions are already training rows) or BIPIA train tables, code attacks into BIPIA train code answers;
- clean twins: every BIPIA train document used, as a benign row.
Documents are at most --max-chars (default 4000) characters, so a planted instruction is never cut off by training's
1,536-token truncation. BIPIA's train and test emails share templates (33 of 50 test emails have a near-copy in
train, e.g. Mercury bank notices), so any train document whose 8-word shingles overlap a test document's by half
or more (guardrate.py audit's rule, either direction) is left out, keeping the BIPIA test split unseen. Validation: run 5's
plus 10% of the attack strings (held out of training) planted the same way, and held-out clean tables.
Known gap (found after run 9; left as is so this script reproduces run 9's data): attacked validation emails come
from the training email pool, and the new clean validation rows are tables only. Run 5's clean emails (aeslc,
hard benign) are in validation, but none in BIPIA's "SUBJECT: ..|EMAIL_FROM: ..|" header format, the format run 9
blocked when clean (experiments.md, "Run 9 results"). A next builder should hold out part of the email pool,
header-format emails included, and validate both their clean and injected versions.
Insertion: build_injection_bench.insert (as BIPIA's insert_* functions).
"""

import argparse
import json
import random
from collections import Counter

from build_injection_bench import SRC, insert
from guardrate import _norm, _shingles

QUESTION_ROW = {"mode": "request", "response": None, "question": "block"}


def run5_rows(path, sources):
    """split_{train,val} records -> build_v6 rows (train_d3.normalise gives the same fields either way)."""
    rows = []
    for rec in map(json.loads, open(path)):
        task = {c["task"]: c["true_label"][0] for c in rec["output"]["classifications"]}
        rows.append({**QUESTION_ROW, "text": rec["input"], "gold_block": task["block"] == "block",
                     "risk": task["risk"], "source": sources[rec["input"]]})
    return rows


def contexts(task, split):
    out = []
    for line in open(f"{SRC}/bipia/{task}/{split}.jsonl"):
        c = json.loads(line)["context"]
        out.append("\n".join(c) if isinstance(c, list) else c)
    return out


def near_test(text, test_shingles):
    """True if text and some test document share half or more of either's 8-word shingles."""
    s = _shingles(_norm(text).split())
    return bool(s) and any(len(s & t) >= 0.5 * min(len(s), len(t)) for t in test_shingles if t)


def bipia_docs(task, max_chars):
    test = [_shingles(_norm(t).split()) for t in contexts(task, "test")]
    docs = sorted({t for t in contexts(task, "train") if len(t) <= max_chars})
    kept = [t for t in docs if not near_test(t, test)]
    print(f"BIPIA {task}: {len(docs)} train documents <= max chars, {len(docs) - len(kept)} near a test document dropped")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", default="data/split_train.jsonl")
    ap.add_argument("--val", default="data/split_val.jsonl")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--text-per-attack", type=int, default=16)
    ap.add_argument("--code-per-attack", type=int, default=8)
    ap.add_argument("--val-attack-share", type=float, default=0.1)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sources = {r["text"]: r["source"] for r in map(json.loads, open("data/train_verified.jsonl"))}
    train, val = run5_rows(args.train, sources), run5_rows(args.val, sources)

    our_emails = sorted({r["text"] for r in train if r["source"] in ("aeslc_emails", "hard_benign_emails")
                         and len(r["text"]) <= args.max_chars})
    emails = bipia_docs("email", args.max_chars)
    tables = bipia_docs("table", args.max_chars)
    code = bipia_docs("code", args.max_chars)
    all_emails, bipia_emails = emails + our_emails, set(emails)
    rng.shuffle(tables)
    val_tables, tables = tables[:50], tables[50:]

    def split_attacks(path):
        strs = sorted(s for v in json.load(open(f"{SRC}/bipia/{path}")).values() for s in v)
        rng.shuffle(strs)
        k = max(1, round(len(strs) * args.val_attack_share))
        return strs[k:], strs[:k]

    text_tr, text_va = split_attacks("text_attack_train.json")
    code_tr, code_va = split_attacks("code_attack_train.json")

    def plant(attacks, per_attack, pick_doc):
        rows, used = [], set()
        for attack in attacks:
            for _ in range(per_attack):
                doc, bipia_doc = pick_doc()
                if bipia_doc:
                    used.add(doc)
                pos = rng.choice(["start", "middle", "end"])
                rows.append({**QUESTION_ROW, "text": insert(doc, attack, pos, rng), "gold_block": True,
                             "risk": "prompt_attack", "source": "planted_injection"})
        return rows, used

    def text_doc(table_pool):
        if rng.random() < 0.5:
            return rng.choice(table_pool), True
        doc = rng.choice(all_emails)  # one pool: the few BIPIA train emails left must not be reused heavily
        return doc, doc in bipia_emails

    new_train, used_tr = plant(text_tr, args.text_per_attack, lambda: text_doc(tables))
    rows, used = plant(code_tr, args.code_per_attack, lambda: (rng.choice(code), True))
    new_train += rows
    used_tr |= used
    new_train += [{**QUESTION_ROW, "text": d, "gold_block": False, "risk": "no_risk", "source": "planted_clean"}
                  for d in sorted(used_tr)]

    new_val, used_va = plant(text_va, 4, lambda: text_doc(val_tables))
    rows, used = plant(code_va, 4, lambda: (rng.choice(code), True))
    new_val += rows
    new_val += [{**QUESTION_ROW, "text": d, "gold_block": False, "risk": "no_risk", "source": "planted_clean"}
                for d in sorted(t for t in used_va | used if t in set(val_tables))]

    train_texts = {r["text"] for r in train + new_train}
    new_val = [r for r in new_val if r["text"] not in train_texts]
    train += new_train
    val += new_val
    rng.shuffle(train)
    for path, rows in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        added = Counter((r["source"], r["gold_block"]) for r in rows if r["source"].startswith("planted"))
        print(f"{len(rows)} rows -> {path}; added: " + ", ".join(f"{s} {'T' if b else 'F'} {n}" for (s, b), n in sorted(added.items())))
    print(f"documents: BIPIA emails {len(emails)}, tables {len(tables)} (+{len(val_tables)} val), code {len(code)}, "
          f"run-5 emails {len(our_emails)}; attack strings train {len(text_tr)}+{len(code_tr)}, val {len(text_va)}+{len(code_va)}")


if __name__ == "__main__":
    main()
