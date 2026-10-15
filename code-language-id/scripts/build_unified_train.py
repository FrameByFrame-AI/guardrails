#!/usr/bin/env python3
"""Build the unified 113-language multi-label TRAIN + DEV corpus (additively).

Layers (all kept, none replaced — the v8 regression came from *replacing*):
  - v1_splits_clean train+val : 107-lang single-label, pure code -> 1-hot rows
                                 (capped per language to limit imbalance)
  - Accuknox train-clean       : gold 25-lang single / multi / benign
  - stack-polyglot             : natural config-format polyglot (the 6 formats)

Then, because the single-label base would dilute multi supervision to ~5 %
(worse than v7's ~18 %), we additively SYNTHESIZE multi rows from the unified
single pool. v1's synthetic multi was *pure-code concatenation* with no prose,
but the real Accuknox multi rows (and the 113-eval) are **prose + code**: a wall
of distractor sentences drawn from a tiny shared pool, then 2-3 code snippets in
different languages joined by short transition phrases ("Then there's this one
in another language:"). Training on pure-code concat taught "multi == two code
blocks back to back" and the model missed secondaries buried in prose (v1 multi
error 51 %, mostly far-miss FN). So we now replicate that construction faithfully
— harvesting the distractor sentences from the real benign rows, mixing a
prose-heavy and a translate-style template, biasing language co-occurrence toward
the Accuknox-25 (what actually co-occurs / what the eval tests) while guaranteeing
every one of the 113 appears as a secondary at least ``--min-secondary`` times.

Every code component is byte-budgeted so it lands inside the model's 2048-byte
window (a truncation that dropped a language would teach the wrong supervision)
and we emit its **byte-offset span** so a future per-position / span head can be
trained on the same corpus for free. Spans are half-open [start, end) byte
ranges into the utf-8 prompt; prose/separators are background (no span). The
``spans`` column is inert for the presence-only trainer (which reads only
``target``); single rows carry ``[]`` (convention: a single-kind row's whole text
is its one language) and benign rows carry ``[]`` (all background).

Outputs (Accuknox schema: prompt, target JSON, kind, + source, + spans JSON):
  train-unified-113-00000-of-00001.parquet
  dev-unified-113-00000-of-00001.parquet      (real rows only; for thresholds)

DEV is a stratified, real-only hold-out — never synthetic — and is disjoint
from both TRAIN and the 113-language test eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_language_id.taxonomy import ACCUKNOX_25, LANGUAGES, resolve  # noqa: E402

MODELS = Path("/models")
V1_SPLITS = MODELS / "guardrail_code_data/processed/v1_splits_clean"
ACCUKNOX = MODELS / "Accuknoxtechnologies-CodeLanguage/data"
EVAL_PARQUET = ACCUKNOX / "eval-unified-113-00000-of-00001.parquet"

SEPARATORS = (
    "\n", "\n\n", "\n\n", "\n\n---\n\n",
    "\n\nAlso, here's another snippet:\n",
    "\n\nAnd separately:\n",
    "\n\nLater in the same file:\n",
    "\n\nFor comparison, consider:\n",
)

# Short prose phrases that real Accuknox rows use to introduce / separate the
# embedded code snippets. Code components are appended after one of these so the
# model learns to find code *after* natural-language framing.
LEAD_INS = (
    "Translate this into another language for me:",
    "Refactor the function below:",
    "Can you convert the following snippet?",
    "Here's some code I'm working with:",
    "Take a look at this and let me know what you think:",
    "I need help understanding this:",
    "Please review the code below:",
    "What does this do?",
)
TRANSITIONS = (
    "Then there's this one in another language:",
    "And separately, there's this:",
    "Also, here's another snippet:",
    "Later in the same file:",
    "For comparison, consider:",
    "Here's a related piece:",
    "And this part too:",
    "Then I also have:",
)

# Fallback distractor sentences, used only if the benign pool can't be harvested.
_FALLBACK_DISTRACTORS = (
    "The technology conference attracted engineers from around the world.",
    "He walked along the river path, listening to the rustling of leaves overhead.",
    "Economic indicators showed a gradual recovery following the downturn.",
)


def norm_hash(text: str) -> str:
    import re
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip()
                          .encode("utf-8", "replace")).hexdigest()


def bucket(text: str) -> float:
    """Stable [0,1) bucket from the prompt hash (for reproducible splits)."""
    return int(norm_hash(text)[:8], 16) / 0xFFFFFFFF


def truncate_bytes(s: str, n: int) -> str:
    return s.encode("utf-8", "replace")[:n].decode("utf-8", "ignore")


def single_target(canon: str) -> str:
    return json.dumps({"is_valid": True, "category": {canon: True}}, ensure_ascii=False)


def reresolve_target(raw: str) -> str:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    cats = {resolve(k): True for k in (obj.get("category") or {})}
    return json.dumps({"is_valid": bool(obj.get("is_valid", bool(cats))),
                       "category": cats}, ensure_ascii=False)


def load_v1(max_per_lang: int, seed: int) -> pd.DataFrame:
    frames = []
    for split in ("train", "val"):
        d = pd.read_parquet(V1_SPLITS / f"{split}.parquet",
                            columns=["code", "canonical_language"])
        d = d.rename(columns={"code": "prompt"})
        d["canon"] = d["canonical_language"].map(lambda x: resolve(x))
        d["source"] = f"v1_{split}"
        frames.append(d[["prompt", "canon", "source"]])
    df = pd.concat(frames, ignore_index=True)
    if max_per_lang > 0:
        parts = [g.sample(n=min(len(g), max_per_lang), random_state=seed)
                 for _, g in df.groupby("canon")]
        df = pd.concat(parts, ignore_index=True)
    df["target"] = df["canon"].map(single_target)
    df["kind"] = "single"
    df["spans"] = "[]"
    return df[["prompt", "target", "kind", "source", "spans"]]


def load_accuknox(name: str, source: str) -> pd.DataFrame:
    df = pd.read_parquet(ACCUKNOX / name)
    df = df.copy()
    df["target"] = df["target"].map(reresolve_target)
    df["source"] = source
    # Real rows carry no span supervision yet (a follow-up adds natural-polyglot
    # spans in extract_polyglot_from_stack.py); empty list = no span labels.
    df["spans"] = "[]"
    return df[["prompt", "target", "kind", "source", "spans"]]


def primary_langs(target: str) -> list[str]:
    return sorted(json.loads(target).get("category") or {})


def harvest_distractors(benign_prompts) -> list[str]:
    """Pull the (tiny, shared) distractor-sentence pool out of the benign rows.

    The real Accuknox prose+code rows pad with sentences drawn from this same
    closed pool (~61 unique sentences), so reusing it makes our synthetic prose
    distributionally identical to the eval's filler.
    """
    pool: set[str] = set()
    for p in benign_prompts:
        if not isinstance(p, str):
            continue
        for s in re.split(r"(?<=[.!?])\s+", p.strip()):
            s = s.strip()
            if 15 <= len(s) <= 240:
                pool.add(s)
    return sorted(pool) or list(_FALLBACK_DISTRACTORS)


def _weighted_sample(pool: list[str], weights: dict[str, float], n: int,
                     rng: random.Random) -> list[str]:
    """Sample n distinct languages with the given per-language weights."""
    remaining = list(pool)
    chosen: list[str] = []
    for _ in range(min(n, len(remaining))):
        w = [weights.get(l, 1.0) for l in remaining]
        pick = rng.choices(remaining, weights=w, k=1)[0]
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _build_multi_prompt(langs: list[str], by_lang: dict[str, list[str]],
                        distractors: list[str], rng: random.Random,
                        code_total: int, prose_budget: int) -> tuple[str, list[dict], list[tuple]]:
    """Compose one prose+code multi prompt and its byte-offset code spans.

    Returns (prompt, spans, key_parts). ``spans`` is a list of
    {"lang", "start", "end"} half-open utf-8 byte ranges for each code snippet;
    ``key_parts`` identifies the snippets chosen (for dedup).
    """
    n = len(langs)
    per = max(80, code_total // n)
    snippets: list[str] = []
    key_parts: list[tuple] = []
    for lang in langs:
        choices = by_lang[lang]
        txt = choices[rng.randrange(len(choices))]
        snippets.append(truncate_bytes(txt, per))
        key_parts.append((lang, norm_hash(txt)[:12]))

    parts: list[str] = []
    spans: list[dict] = []
    blen = 0  # running utf-8 byte length of what's been appended

    def add(text: str, lang: str | None = None) -> None:
        nonlocal blen
        b = len(text.encode("utf-8", "replace"))
        if lang is not None:
            spans.append({"lang": lang, "start": blen, "end": blen + b})
        parts.append(text)
        blen += b

    style = rng.choices(("prose_heavy", "translate", "plain"),
                        weights=(55, 30, 15), k=1)[0]

    if style == "prose_heavy":
        sents, budget = [], prose_budget
        while budget > 0:
            s = rng.choice(distractors)
            cost = len(s.encode("utf-8", "replace")) + 1
            if cost > budget:
                break
            sents.append(s)
            budget -= cost
        if sents:
            add(" ".join(sents) + "\n\n")
        for i, (lang, snip) in enumerate(zip(langs, snippets)):
            if i > 0:
                add("\n\n" + rng.choice(TRANSITIONS) + "\n")
            add(snip, lang=lang)
    elif style == "translate":
        add(rng.choice(LEAD_INS) + "\n")
        for i, (lang, snip) in enumerate(zip(langs, snippets)):
            if i > 0:
                add("\n\n" + rng.choice(TRANSITIONS) + "\n")
            add(snip, lang=lang)
    else:  # plain — bare concatenation (some real rows have no prose framing)
        for i, (lang, snip) in enumerate(zip(langs, snippets)):
            if i > 0:
                add(rng.choice(SEPARATORS))
            add(snip, lang=lang)

    return "".join(parts), spans, key_parts


def synthesize_multi(singles: pd.DataFrame, distractors: list[str], n_target: int,
                     three_way_frac: float, max_bytes: int, min_secondary: int,
                     prose_budget: int, eval_hashes: set, seed: int,
                     max_factor: int = 12) -> list[dict]:
    """Synthesize realistic prose+code multi rows with byte-offset code spans.

    Two passes: (1) a *guarantee* pass that gives every language with available
    single snippets at least ``min_secondary`` appearances as a multi label, then
    (2) a *fill* pass with Accuknox-25-biased weighted sampling to reach
    ``n_target``. Both share a dedup set keyed on the (langs, snippet-hashes).
    """
    by_lang: dict[str, list[str]] = defaultdict(list)
    for prompt, target in zip(singles["prompt"], singles["target"]):
        langs = primary_langs(target)
        if len(langs) == 1 and isinstance(prompt, str):
            by_lang[langs[0]].append(prompt)
    pool = [l for l in LANGUAGES if by_lang.get(l)]
    # Bias co-occurrence toward the 25 Accuknox langs (what actually co-occurs in
    # the real data and what the 113-eval tests), tail langs kept at base weight.
    weights = {l: (4.0 if l in set(ACCUKNOX_25) else 1.0) for l in pool}

    rng = random.Random(seed)
    seen: set = set()
    rows: list[dict] = []

    def emit(langs: list[str]) -> bool:
        n = len(langs)
        code_total = max(n * 80, max_bytes - prose_budget)
        prompt, spans, key_parts = _build_multi_prompt(
            langs, by_lang, distractors, rng, code_total, prose_budget)
        key = tuple(sorted(key_parts))
        if key in seen or norm_hash(prompt) in eval_hashes:
            return False
        seen.add(key)
        cats = {l: True for l in langs}
        rows.append({
            "prompt": prompt,
            "target": json.dumps({"is_valid": True, "category": cats},
                                  ensure_ascii=False),
            "kind": "multi",
            "source": f"synth_{n}way",
            "spans": json.dumps(spans, ensure_ascii=False),
        })
        return True

    # ---- pass 1: guarantee per-language secondary coverage -------------------
    for primary in pool:
        made, attempts = 0, 0
        while made < min_secondary and attempts < min_secondary * max_factor:
            attempts += 1
            n = 3 if (rng.random() < three_way_frac and len(pool) >= 3) else 2
            partners = _weighted_sample([l for l in pool if l != primary],
                                        weights, n - 1, rng)
            if len(partners) < n - 1:
                break
            langs = [primary] + partners
            rng.shuffle(langs)
            if emit(langs):
                made += 1

    # ---- pass 2: fill to the global target with weighted sampling ------------
    attempts = 0
    while len(rows) < n_target and attempts < n_target * max_factor:
        attempts += 1
        n = 3 if (rng.random() < three_way_frac and len(pool) >= 3) else 2
        langs = _weighted_sample(pool, weights, n, rng)
        if len(langs) < n:
            break
        emit(langs)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", type=Path,
                    default=ACCUKNOX / "train-unified-113-00000-of-00001.parquet")
    ap.add_argument("--out-dev", type=Path,
                    default=ACCUKNOX / "dev-unified-113-00000-of-00001.parquet")
    ap.add_argument("--max-single-per-lang", type=int, default=500)
    ap.add_argument("--dev-frac", type=float, default=0.12)
    ap.add_argument("--multi-target-frac", type=float, default=0.30)
    ap.add_argument("--three-way-frac", type=float, default=0.24)
    # 2048-byte model window: keep the whole prompt inside it so every code span
    # stays visible. ~1300B code + ~550B prose leaves margin under 2048.
    ap.add_argument("--max-multi-bytes", type=int, default=1850)
    ap.add_argument("--prose-budget", type=int, default=550)
    ap.add_argument("--min-secondary", type=int, default=40,
                    help="Guaranteed multi appearances per language (validator wants >=30).")
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    # 1. load + normalize all real sources -----------------------------------
    real = pd.concat([
        load_v1(args.max_single_per_lang, args.seed),
        load_accuknox("train-clean-00000-of-00001.parquet", "accuknox_train_clean"),
        load_accuknox("stack-polyglot-00000-of-00001.parquet", "stack_polyglot"),
    ], ignore_index=True)
    print(f"real rows loaded = {len(real):,}  by source={dict(Counter(real['source']))}")

    # 2. dedup + drop anything overlapping the 113-eval -----------------------
    real["h"] = real["prompt"].map(norm_hash)
    before = len(real)
    real = real.drop_duplicates(subset="h", keep="first")
    print(f"internal dup rows dropped = {before - len(real):,}")

    eval_hashes = set(pd.read_parquet(EVAL_PARQUET, columns=["prompt"])["prompt"]
                      .map(norm_hash))
    overlap = real["h"].isin(eval_hashes)
    print(f"rows overlapping 113-eval dropped = {int(overlap.sum()):,}")
    real = real[~overlap].reset_index(drop=True)
    assert not real["h"].isin(eval_hashes).any()

    # 3. stratified real-only dev split (per kind, stable hash) ---------------
    b = real["prompt"].map(bucket)
    is_dev = b < args.dev_frac
    dev = real[is_dev].drop(columns="h").reset_index(drop=True)
    train_real = real[~is_dev].reset_index(drop=True)
    print(f"\ndev rows  = {len(dev):,}  kinds={dict(Counter(dev['kind']))}")
    print(f"train_real= {len(train_real):,}  kinds={dict(Counter(train_real['kind']))}")

    # 4. synthesize multi into TRAIN to hit the target multi fraction ---------
    counts = Counter(train_real["kind"])
    S, M, B = counts["single"], counts["multi"], counts["benign"]
    f = args.multi_target_frac
    need = max(0, int(round((f * (S + M + B) - M) / (1 - f))))
    print(f"\nsynthesizing {need:,} multi rows to reach multi_frac={f:.2f} "
          f"(have S={S:,} M={M:,} B={B:,})")
    distractors = harvest_distractors(train_real[train_real["kind"] == "benign"]["prompt"])
    print(f"  harvested {len(distractors)} distractor sentences from benign rows")
    synth = synthesize_multi(
        train_real.drop(columns="h"), distractors, need,
        args.three_way_frac, args.max_multi_bytes, args.min_secondary,
        args.prose_budget, eval_hashes, args.seed)
    print(f"  emitted {len(synth):,} synthetic multi rows")

    train = pd.concat([train_real.drop(columns="h"), pd.DataFrame(synth)],
                      ignore_index=True)
    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 5. report + write -------------------------------------------------------
    tc = Counter(train["kind"]); tot = len(train)
    print(f"\nFINAL train = {tot:,} rows")
    for k in ("single", "multi", "benign"):
        print(f"  {k:8} {tc[k]:>7} ({100*tc[k]/tot:.1f}%)")
    print(f"  by source={dict(Counter(train['source']))}")

    lang_cov = Counter()
    for t in train["target"]:
        for l in primary_langs(t):
            lang_cov[l] += 1
    missing = [l for l in LANGUAGES if lang_cov[l] == 0]
    print(f"languages covered in train = {len(LANGUAGES)-len(missing)}/{len(LANGUAGES)}")
    if missing:
        print(f"  MISSING: {missing}")

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out_train, index=False)
    dev.to_parquet(args.out_dev, index=False)
    print(f"\nwrote {args.out_train}")
    print(f"wrote {args.out_dev}")


if __name__ == "__main__":
    main()
