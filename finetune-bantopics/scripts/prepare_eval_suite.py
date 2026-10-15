#!/usr/bin/env python3
"""Build the comprehensive eval suite for BanTopics.

Output JSONL row schema (compatible with ``benchmark_bantopics.py`` and the
in-training eval in ``finetune_lora.py``):

    {
      "prompt": str,
      "expected": "BLOCK" | "PASS",
      "topics": list[str],          # what the scanner is configured with
      "language": "en" | "zh" | ...,
      "eval_section": str,          # see SECTIONS below
      "source_id": str
    }

Sections (each scored independently — production_50 is the headline):
  production_50           — current 50-prompt benchmark (40 BLOCK / 10 PASS)
  xstest_safe             — XSTest "safe" prompts (PASS — over-refusal regression)
  xstest_unsafe           — XSTest "unsafe" prompts (BLOCK)
  harmbench_holdout       — held-out HarmBench (BLOCK)
  jbb_holdout             — held-out JBB-Behaviors (BLOCK)
  dna_holdout             — held-out do-not-answer (BLOCK)
  safetybench_zh          — Chinese SafetyBench questions (BLOCK; multilingual transfer signal)

The held-out splits use the same hash-based split as ``prepare_pairs.py`` so a
row never appears in both training and eval.

    docker compose run --rm prepare-eval-suite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_bantopics.splits import is_held_out
from finetune_bantopics.topic_mappings import (
    DNA_TYPES_OF_HARM_TO_TOPICS,
    HARMBENCH_CATEGORY_TO_TOPICS,
    JBB_CATEGORY_TO_TOPICS,
    SAFETYBENCH_CATEGORY_TO_TOPICS,
    XSTEST_CATEGORY_TO_TOPICS,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------

def section_production_50(args: argparse.Namespace) -> Iterable[dict]:
    """Pull the existing 50-prompt benchmark fixture as-is."""
    if not args.production_eval_jsonl.exists():
        print(f"# production_50: skip (run extract-bantopics-eval first)", file=sys.stderr)
        return
    with open(args.production_eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield {
                "prompt": row["prompt"],
                "expected": row["expected"],
                "topics": row["topics"],
                "language": row.get("language", "en"),
                "eval_section": "production_50",
                "source_id": row.get("row_id", ""),
            }


def section_xstest(args: argparse.Namespace) -> Iterable[dict]:
    """XSTest schema: ``label`` is "safe"/"unsafe", ``type`` is the harm
    category (homonyms, figurative_language, ...), ``note`` sometimes carries
    the explicit semantic topic (violence, drugs). We resolve topics by:

       1. ``note`` if it directly matches a trained topic (most precise)
       2. ``type`` via ``XSTEST_CATEGORY_TO_TOPICS`` (covers all 18 type values)
    """
    from datasets import load_dataset
    from finetune_bantopics.topics import all_topics as _all_topics

    trained_topics = set(_all_topics())
    ds = load_dataset("walledai/XSTest", split="test")
    print(f"# loaded XSTest: {len(ds)} rows", file=sys.stderr)

    skipped_no_topic = 0
    for i, row in enumerate(ds):
        label = (row.get("label") or "").lower()
        type_ = row.get("type") or ""
        note = (row.get("note") or "").strip().lower()
        prompt = row.get("prompt") or ""

        topics: list[str] = []
        if note and note != "none" and note in trained_topics:
            topics = [note]
        if not topics:
            topics = list(XSTEST_CATEGORY_TO_TOPICS.get(type_, []))
        if not topics:
            skipped_no_topic += 1
            continue

        if label == "safe":
            section, expected = "xstest_safe", "PASS"
        elif label == "unsafe":
            section, expected = "xstest_unsafe", "BLOCK"
        else:
            continue

        yield {
            "prompt": prompt,
            "expected": expected,
            "topics": topics,
            "language": "en",
            "eval_section": section,
            "source_id": f"xstest:{i}",
        }
    if skipped_no_topic:
        print(f"# xstest: skipped {skipped_no_topic} rows with unmappable type",
              file=sys.stderr)


def section_harmbench(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    print(f"# loaded HarmBench: {len(ds)} rows", file=sys.stderr)
    for i, row in enumerate(ds):
        category = row.get("category") or row.get("functional_category") or ""
        topics = HARMBENCH_CATEGORY_TO_TOPICS.get(category, [])
        if not topics:
            continue
        row_id = str(row.get("id") or i)
        if not is_held_out("harmbench:standard", row_id, args.holdout_pct):
            continue
        yield {
            "prompt": row.get("prompt") or row.get("behavior") or "",
            "expected": "BLOCK",
            "topics": topics,
            "language": "en",
            "eval_section": "harmbench_holdout",
            "source_id": row_id,
        }


def section_jbb(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    print(f"# loaded JBB-Behaviors: {len(ds)} rows", file=sys.stderr)
    for i, row in enumerate(ds):
        category = row.get("Category") or row.get("category") or ""
        topics = JBB_CATEGORY_TO_TOPICS.get(category, [])
        if not topics:
            continue
        row_id = str(row.get("Index") or i)
        if not is_held_out("jbb:harmful", row_id, args.holdout_pct):
            continue
        yield {
            "prompt": row.get("Goal") or row.get("goal") or row.get("prompt") or "",
            "expected": "BLOCK",
            "topics": topics,
            "language": "en",
            "eval_section": "jbb_holdout",
            "source_id": row_id,
        }


def section_dna(args: argparse.Namespace) -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("LibrAI/do-not-answer", split="train")
    print(f"# loaded do-not-answer: {len(ds)} rows", file=sys.stderr)
    for i, row in enumerate(ds):
        types_of_harm = row.get("types_of_harm") or ""
        topics = DNA_TYPES_OF_HARM_TO_TOPICS.get(types_of_harm, [])
        if not topics:
            continue
        row_id = str(row.get("id") or i)
        if not is_held_out("dna", row_id, args.holdout_pct):
            continue
        yield {
            "prompt": row.get("question") or "",
            "expected": "BLOCK",
            "topics": topics,
            "language": "en",
            "eval_section": "dna_holdout",
            "source_id": row_id,
        }


def section_safetybench_zh(args: argparse.Namespace) -> Iterable[dict]:
    """SafetyBench Chinese — pulls the question text only, treats expected
    as BLOCK (these are MCQs about harmful scenarios). Format mismatch noted
    in README; primary use is multilingual encoder transfer signal."""
    from datasets import load_dataset

    try:
        ds = load_dataset("thu-coai/SafetyBench", "zh", split="test")
    except Exception as exc:
        print(f"# safetybench_zh: skip ({exc})", file=sys.stderr)
        return
    print(f"# loaded SafetyBench-zh: {len(ds)} rows", file=sys.stderr)
    cap = args.safetybench_zh_cap
    for i, row in enumerate(ds):
        if cap and i >= cap:
            break
        category = row.get("category") or ""
        topics = SAFETYBENCH_CATEGORY_TO_TOPICS.get(category, [])
        if not topics:
            continue
        question = row.get("question") or ""
        if not question.strip():
            continue
        yield {
            "prompt": question,
            "expected": "BLOCK",
            "topics": topics,
            "language": "zh",
            "eval_section": "safetybench_zh",
            "source_id": str(row.get("id") or i),
        }


SECTION_BUILDERS = [
    section_production_50,
    section_xstest,
    section_harmbench,
    section_jbb,
    section_dna,
    section_safetybench_zh,
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--production-eval-jsonl", type=Path,
                   default=Path("/workspace/data/eval/bantopics_50.jsonl"))
    p.add_argument("--holdout-pct", type=int, default=10,
                   help="Same value used by prepare_pairs.py --exclude-holdout-pct.")
    p.add_argument("--safetybench-zh-cap", type=int, default=300,
                   help="Cap SafetyBench-zh rows to keep the eval suite small.")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    section_counts: dict[str, int] = {}
    expected_counts: dict[str, dict[str, int]] = {}
    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for builder in SECTION_BUILDERS:
            for row in builder(args):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                section_counts[row["eval_section"]] = section_counts.get(row["eval_section"], 0) + 1
                ec = expected_counts.setdefault(row["eval_section"], {"BLOCK": 0, "PASS": 0})
                ec[row["expected"]] = ec.get(row["expected"], 0) + 1
                n += 1

    print(f"\n# wrote {args.output}: {n} eval rows across {len(section_counts)} sections")
    for sec, count in section_counts.items():
        ec = expected_counts[sec]
        print(f"  {count:5d}  {sec:<25s}  BLOCK={ec.get('BLOCK', 0):4d}  PASS={ec.get('PASS', 0):4d}")


if __name__ == "__main__":
    main()
