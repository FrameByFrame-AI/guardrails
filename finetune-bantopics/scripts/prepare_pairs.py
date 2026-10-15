#!/usr/bin/env python3
"""Unified (sentence, topic) pair extractor for BanTopics training.

One adapter per source dataset; each yields records of the shape

    {"sentence": str, "topic": str, "source": str, "row_id": str}

then they all hit the same write path with the same leakage filter and
sentence-length gates. New sources are added by registering an adapter in
``ADAPTERS`` and (where needed) a category mapping in
``finetune_bantopics.topic_mappings``.

    docker compose run --rm prepare-pairs                        # uses BT_PAIR_SOURCE
    python scripts/prepare_pairs.py --source beavertails ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.splits import is_held_out
from finetune_bantopics.splitter import split_sentences
from finetune_bantopics.topic_mappings import (
    AEGIS_CATEGORY_TO_TOPICS,
    BEAVERTAILS_FLAG_TO_TOPICS,
    DNA_TYPES_OF_HARM_TO_TOPICS,
    HARMBENCH_CATEGORY_TO_TOPICS,
    JBB_CATEGORY_TO_TOPICS,
    OPENAI_MOD_FLAG_TO_TOPICS,
    WILDGUARD_CATEGORY_TO_TOPICS,
)


# ---------------------------------------------------------------------------
# leakage filter — drop training sentences that overlap an eval 8-gram
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_leakage_filter(eval_path: Path | None) -> set[str]:
    if eval_path is None or not eval_path.exists():
        return set()
    ngrams: set[str] = set()
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = _tokenize(row.get("prompt", ""))
            for i in range(len(tokens) - 7):
                ngrams.add(" ".join(tokens[i:i + 8]))
    return ngrams


def overlaps(sentence: str, eval_ngrams: set[str]) -> bool:
    if not eval_ngrams:
        return False
    tokens = _tokenize(sentence)
    if len(tokens) < 8:
        return False
    for i in range(len(tokens) - 7):
        if " ".join(tokens[i:i + 8]) in eval_ngrams:
            return True
    return False


# ---------------------------------------------------------------------------
# adapters — each yields (sentence, topic, source, row_id) records
# ---------------------------------------------------------------------------

def _split_or_keep(prompt: str, language: str, backend: str) -> list[str]:
    sents = split_sentences(prompt, language=language, backend=backend) or [prompt]
    return [s.strip() for s in sents if s and s.strip()]


def _emit(prompt: str, topics: list[str], source: str, row_id: str,
          language: str, backend: str) -> Iterable[dict]:
    """Yield (sentence, topic) records for a prompt mapped to >=1 topic."""
    if not topics:
        return
    for sent in _split_or_keep(prompt, language=language, backend=backend):
        for t in dict.fromkeys(topics):
            yield {
                "sentence": sent,
                "topic": t,
                "source": source,
                "row_id": row_id,
            }


# ---------- aegis ----------

def adapt_aegis(args: argparse.Namespace) -> Iterable[dict]:
    aegis_dir = _resolve_hf_snapshot(args.aegis_dir, "train.json")
    print(f"# aegis snapshot: {aegis_dir}")
    for split in ("train", "validation"):
        path = aegis_dir / f"{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        print(f"# loaded aegis/{split}: {len(data)} rows")
        for row in data:
            if row.get("prompt_label") != "unsafe":
                continue
            cats = [c.strip() for c in (row.get("violated_categories") or "").split(",") if c.strip()]
            topics: list[str] = []
            for c in cats:
                topics.extend(AEGIS_CATEGORY_TO_TOPICS.get(c, []))
            yield from _emit(
                prompt=row.get("prompt") or "",
                topics=topics,
                source=f"aegis:{split}",
                row_id=str(row.get("id")),
                language=args.splitter_language,
                backend=args.splitter_backend,
            )


# ---------- beavertails ----------

def adapt_beavertails(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "PKU-Alignment/BeaverTails",
        split=f"30k_train{f'[:{args.max_rows}]' if args.max_rows and args.max_rows > 0 else ''}",
    )
    print(f"# loaded beavertails: {len(ds)} rows")
    for i, row in enumerate(ds):
        if not row.get("is_safe", True) is False:  # only unsafe rows
            pass
        # Schema: row['category'] is a dict of 14 booleans.
        cat_flags = row.get("category", {}) or {}
        topics: list[str] = []
        for flag, on in cat_flags.items():
            if on:
                topics.extend(BEAVERTAILS_FLAG_TO_TOPICS.get(flag, []))
        if not topics:
            continue
        yield from _emit(
            prompt=row.get("prompt") or "",
            topics=topics,
            source="beavertails:30k_train",
            row_id=str(i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


# ---------- harmbench ----------

def adapt_harmbench(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    print(f"# loaded harmbench/standard: {len(ds)} rows")
    for i, row in enumerate(ds):
        category = row.get("category") or row.get("functional_category") or ""
        topics = HARMBENCH_CATEGORY_TO_TOPICS.get(category, [])
        prompt = row.get("prompt") or row.get("behavior") or ""
        yield from _emit(
            prompt=prompt,
            topics=topics,
            source="harmbench:standard",
            row_id=str(row.get("id") or i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


# ---------- jbb ----------

def adapt_jbb(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    print(f"# loaded jbb/behaviors/harmful: {len(ds)} rows")
    for i, row in enumerate(ds):
        category = row.get("Category") or row.get("category") or ""
        topics = JBB_CATEGORY_TO_TOPICS.get(category, [])
        prompt = row.get("Goal") or row.get("goal") or row.get("prompt") or ""
        yield from _emit(
            prompt=prompt,
            topics=topics,
            source="jbb:harmful",
            row_id=str(row.get("Index") or i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


# ---------- do-not-answer ----------

def adapt_dna(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("LibrAI/do-not-answer", split="train")
    print(f"# loaded do-not-answer: {len(ds)} rows")
    for i, row in enumerate(ds):
        types_of_harm = row.get("types_of_harm") or ""
        topics = DNA_TYPES_OF_HARM_TO_TOPICS.get(types_of_harm, [])
        prompt = row.get("question") or ""
        yield from _emit(
            prompt=prompt,
            topics=topics,
            source="dna",
            row_id=str(row.get("id") or i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


# ---------- wildguardmix ----------

def adapt_wildguard(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "allenai/wildguardmix", "wildguardtrain",
        split=f"train{f'[:{args.max_rows}]' if args.max_rows and args.max_rows > 0 else ''}",
    )
    print(f"# loaded wildguardmix: {len(ds)} rows")
    for i, row in enumerate(ds):
        if row.get("prompt_harm_label") != "harmful":
            continue
        sub = row.get("subcategory") or row.get("prompt_harm_category") or ""
        topics = WILDGUARD_CATEGORY_TO_TOPICS.get(sub, [])
        if not topics:
            continue
        yield from _emit(
            prompt=row.get("prompt") or "",
            topics=topics,
            source="wildguardmix",
            row_id=str(i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


# ---------- openai-moderation-dataset ----------

def adapt_openai_mod(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("walledai/openai-moderation-dataset", split="train")
    print(f"# loaded openai-moderation: {len(ds)} rows")
    for i, row in enumerate(ds):
        topics: list[str] = []
        for flag, mapped in OPENAI_MOD_FLAG_TO_TOPICS.items():
            if row.get(flag):
                topics.extend(mapped)
        if not topics:
            continue
        yield from _emit(
            prompt=row.get("prompt") or "",
            topics=topics,
            source="openai-moderation",
            row_id=str(i),
            language=args.splitter_language,
            backend=args.splitter_backend,
        )


ADAPTERS: dict[str, Callable[[argparse.Namespace], Iterable[dict]]] = {
    "aegis": adapt_aegis,
    "beavertails": adapt_beavertails,
    "harmbench": adapt_harmbench,
    "jbb": adapt_jbb,
    "dna": adapt_dna,
    "wildguard": adapt_wildguard,
    "openai_mod": adapt_openai_mod,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_hf_snapshot(path: Path, marker_filename: str) -> Path:
    """For HF cache layouts: accept either the snapshot dir directly, or the
    'datasets--*' / 'models--*' parent and resolve the latest snapshot."""
    if (path / marker_filename).exists():
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (d for d in snapshots.iterdir() if (d / marker_filename).exists()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise SystemExit(f"could not locate {marker_filename} under {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--eval-jsonl", type=Path, default=None,
                   help="Held-out eval JSONL — sentences sharing an 8-gram are dropped.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Cap rows from the source dataset (0 = no cap; useful for wildguard, beavertails).")
    p.add_argument("--exclude-holdout-pct", type=int, default=10,
                   help="Skip rows whose hash falls in the eval holdout slice (default 10%%).")
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--splitter-language", default="en")
    p.add_argument("--splitter-backend", default="pysbd")
    # aegis-specific
    p.add_argument("--aegis-dir", type=Path, default=None,
                   help="Aegis-only: snapshot dir or HF cache root.")
    args = p.parse_args()

    if args.source == "aegis" and args.aegis_dir is None:
        raise SystemExit("aegis source needs --aegis-dir")

    eval_ngrams = build_leakage_filter(args.eval_jsonl)
    if eval_ngrams:
        print(f"# leakage filter: {len(eval_ngrams)} eval 8-grams loaded")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    adapter = ADAPTERS[args.source]
    pair_count: Counter = Counter()
    skipped: Counter = Counter()
    n_emitted = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for record in adapter(args):
            if is_held_out(record["source"], record["row_id"], args.exclude_holdout_pct):
                skipped["held_out_for_eval"] += 1
                continue
            sent = record["sentence"]
            if len(sent) < args.min_chars:
                skipped["sentence_too_short"] += 1
                continue
            if len(sent) > args.max_chars:
                skipped["sentence_too_long"] += 1
                continue
            if overlaps(sent, eval_ngrams):
                skipped["leaked_to_eval"] += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pair_count[record["topic"]] += 1
            n_emitted += 1

    print(f"\n# wrote {args.output}: {n_emitted} pairs across {len(pair_count)} topics")
    for t, n in pair_count.most_common():
        print(f"  {n:6d}  {t}")
    if skipped:
        print("\n# skipped:")
        for k, n in skipped.most_common():
            print(f"  {n:6d}  {k}")


if __name__ == "__main__":
    main()
