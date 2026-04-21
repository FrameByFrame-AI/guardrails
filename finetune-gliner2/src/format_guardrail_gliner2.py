#!/usr/bin/env python3
"""
Convert all processed datasets into GLiNER2 multi-task guardrail-gliner2 format.

Output per line (GLiNER2 training format):
{
  "input": "<query text>",
  "output": {
    "entities": {},
    "classifications": [
      {"task": "safety",      "labels": [...], "true_label": "unsafe"},
      {"task": "adversarial", "labels": [...], "true_label": ["prompt_injection"]},
      {"task": "harmful",     "labels": [...], "true_label": ["violence"]}
    ]
  }
}

Pure guard model: PII detection moved to a separate NER model (gliner2-multi-korean-pii-lora). Mixing NER into the shared encoder diluted adversarial signal in v3.

Usage:
    python format_guardrail_gliner2.py --input-dir /data/processed --output /data/guardrail_gliner2_train.jsonl
    python format_guardrail_gliner2.py --input-dir /data/processed --output /data/guardrail_gliner2_train.jsonl --max-per-dataset 10000
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


SEED = 42

# DeBERTa-v3 uses relative position encoding — no hard 512 ceiling, but quality
# degrades beyond trained positions. Data stats: p95=413, p99=618, p99.5=896.
# We allow up to 512 total (input + schema) for the DeBERTa sweet spot.
# Records beyond this are either truncated (classification-only) or dropped (NER).
MAX_TOTAL_TOKENS = 512
SPECIAL_TOKENS_BUDGET = 20
MIN_TRUNCATED_INPUT_TOKENS = 64
TOKENIZER_NAME = "fastino/gliner2-multi-v1"

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer

def token_len(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    token_ids = get_tokenizer().encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return get_tokenizer().decode(
        token_ids[:max_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

# ============================================================
# Classification task labels
# ============================================================

SAFETY_LABELS = ["safe", "unsafe"]

ADVERSARIAL_LABELS = [
    "jailbreak",
    "prompt_injection",
    "indirect_injection",
    "instruction_override",
    "data_exfiltration",
    "none",
]

HARMFUL_LABELS = [
    "violence",
    "criminal_planning",
    "hate_speech",
    "harassment",
    "sexual_content",
    "child_exploitation",
    "self_harm",
    "weapons",
    "drugs",
    "profanity",
    "fraud",
    "misinformation",
    "malware",
    "unauthorized_advice",
    "copyright",
    "none",
    # pii_exposure removed: PII detection is handled by entity extraction head.
    # Coupling PII into the harmful head caused the v1 training collapse.
]

# Intent and tone removed from v1: they were derived heuristically (not
# independently annotated), added noise, and amplified upstream label coupling.
# Re-add once we have proper annotation or a teacher model for these tasks.

# ============================================================
# Topic → task mapping
# ============================================================

# Map raw topics from our datasets to adversarial/harmful labels
TOPIC_TO_ADVERSARIAL = {
    "jailbreak": "jailbreak",
    "jailbreaking": "jailbreak",
    "prompt_injection": "prompt_injection",
    "indirect_injection": "indirect_injection",
    # social_engineering dropped: only ~30 train / 6 test examples, unusable.
    # The closest adversarial attack class is prompt_injection.
    "manipulation": "prompt_injection",
}

TOPIC_TO_HARMFUL = {
    # --- Aegis / English canonical ---
    "violence": "violence",
    "criminal_planning": "criminal_planning",
    "hate_speech": "hate_speech",
    "harassment": "harassment",
    "sexual_content": "sexual_content",
    "sexual": "sexual_content",
    "child_exploitation": "child_exploitation",
    "self_harm": "self_harm",
    "weapons": "weapons",
    "drugs": "drugs",
    "profanity": "profanity",
    "fraud": "fraud",
    "misinformation": "misinformation",
    "malware": "malware",
    "unauthorized_advice": "unauthorized_advice",
    "copyright": "copyright",
    "privacy": "pii_exposure",
    "harmful_content": "violence",
    "illegal": "criminal_planning",
    "needs_caution": "harassment",

    # --- Korean moderation (selectstar SELECTSTAR_COLS values) ---
    "insult": "harassment",
    "obscenity": "sexual_content",
    "occupation": "hate_speech",

    # --- Korean normalized topics (from format_training_data.py TOPIC_MAP) ---
    "politics": "hate_speech",
    "age": "hate_speech",
    "race": "hate_speech",
    "religion": "hate_speech",
    "origin": "hate_speech",
    "appearance": "hate_speech",
    "gender": "hate_speech",
    "lgbtq": "hate_speech",
    "nationality": "hate_speech",
    "disability": "hate_speech",
    "education": "hate_speech",
    "personal_attack": "harassment",
    "bias": "hate_speech",
    "other": "harassment",

    # --- Raw Korean topic names (KMHaS, KOLD, korean_unsmile, etc.) ---
    "혐오욕설": "hate_speech",
    "정치성향차별": "hate_speech",
    "출신차별": "hate_speech",
    "외모차별": "hate_speech",
    "성차별": "hate_speech",
    "연령차별": "hate_speech",
    "종교차별": "hate_speech",
    "인종차별": "hate_speech",
    "욕설": "profanity",
    "악플/욕설": "profanity",
    "인종/국적": "hate_speech",
    "여성/가족": "hate_speech",
    "남성": "hate_speech",
    "성소수자": "hate_speech",
    "종교": "hate_speech",
    "지역": "hate_speech",
    "연령": "hate_speech",
    "기타 혐오": "hate_speech",
    "개인지칭": "harassment",

    # --- APEACH English topic names ---
    "Appearance": "hate_speech",
    "Gender sterotypes": "hate_speech",
    "Nationality": "hate_speech",
    "Age and social status": "hate_speech",
    "Sexual harassment": "sexual_content",
    "Education": "hate_speech",
    "Origin and residence": "hate_speech",
    "Disabled": "hate_speech",
    "Racism": "hate_speech",
    "Diets": "harassment",
}

# Adversarial topics that should NOT also appear in harmful (they're attack techniques,
# not harmful content). When a blocked record has ONLY adversarial topics, harmful
# stays ["none"] — that's correct.
ADVERSARIAL_ONLY_TOPICS = {
    "prompt_injection", "jailbreak", "jailbreaking", "indirect_injection",
    "manipulation", "prompt-injection",
}


def classify_record(record):
    """Derive all classification labels from a processed record."""
    blocked = record.get("blocked", False)
    guardrail_type = record.get("type", "")
    raw_topics = record.get("topic", [])

    # Safety
    safety = "unsafe" if blocked else "safe"

    # Adversarial
    adversarial = []
    for t in raw_topics:
        mapped = TOPIC_TO_ADVERSARIAL.get(t)
        if mapped and mapped not in adversarial:
            adversarial.append(mapped)
    # Default for safety-classifier blocked without specific adversarial topic.
    # Many injection datasets (PIGuard, llm-red-teaming) have empty topic lists
    # but the records are clearly adversarial attacks.
    if not adversarial and blocked and guardrail_type == "safety-classifier":
        if any(t in raw_topics for t in ["jailbreak", "jailbreaking"]):
            adversarial = ["jailbreak"]
        elif any(t in raw_topics for t in ["prompt_injection", "prompt-injection"]):
            adversarial = ["prompt_injection"]
        elif any(t in raw_topics for t in ["indirect_injection"]):
            adversarial = ["indirect_injection"]
        else:
            adversarial = ["prompt_injection"]
    if not adversarial:
        adversarial = ["none"]

    # Harmful — decoupled from PII. PII detection is handled by the entity
    # head; forcing pii_exposure into the harmful head created a dominant signal
    # that swamped other categories. PII records now get harmful=["none"] unless
    # they have an explicit harmful topic (e.g. privacy violation in Aegis).
    harmful = []
    for t in raw_topics:
        if t in ADVERSARIAL_ONLY_TOPICS:
            continue
        mapped = TOPIC_TO_HARMFUL.get(t)
        if mapped and mapped != "pii_exposure" and mapped not in harmful:
            harmful.append(mapped)
    # Fallback: blocked records with no mapped harmful topic.
    if not harmful and blocked:
        if guardrail_type == "moderation":
            harmful = ["harassment"]
        elif guardrail_type == "rules-based-protections":
            harmful = ["profanity"]
        elif guardrail_type == "output-validation":
            harmful = ["misinformation"]
        else:
            harmful = ["none"]
    elif not harmful:
        harmful = ["none"]

    return safety, adversarial, harmful


def schema_token_len(output: dict) -> int:
    """Estimate GLiNER2 schema tokens prepended for this record."""
    parts = []
    for cls in output.get("classifications", []):
        parts.append(cls["task"])
        parts.extend(cls["labels"])
    return token_len(" ".join(parts))


def convert_record(record):
    """Convert a processed record to GLiNER2 guardrail-gliner2 format."""
    query = (record.get("query") or "").strip()
    if not query:
        return None, {"reason": "empty_query"}

    safety, adversarial, harmful = classify_record(record)

    # pii-filter records are kept: safety/harmful labels are still valid signal
    # even without the entity head. The entity field is always empty in this v4
    # guard-only model.

    output = {
        "entities": {},
        "classifications": [
            {
                "task": "safety",
                "labels": SAFETY_LABELS,
                "true_label": safety,
            },
            {
                "task": "adversarial",
                "labels": ADVERSARIAL_LABELS,
                "true_label": adversarial,
                "multi_label": True,
            },
            {
                "task": "harmful",
                "labels": HARMFUL_LABELS,
                "true_label": harmful,
                "multi_label": True,
            },
        ],
    }

    schema_tokens = schema_token_len(output)
    available_input_tokens = (
        MAX_TOTAL_TOKENS - schema_tokens - SPECIAL_TOKENS_BUDGET
    )
    if available_input_tokens < MIN_TRUNCATED_INPUT_TOKENS:
        return None, {
            "reason": "schema_budget_too_small",
            "schema_tokens": schema_tokens,
        }

    if token_len(query) > available_input_tokens:
        query = truncate_to_tokens(query, available_input_tokens)
        if token_len(query) < MIN_TRUNCATED_INPUT_TOKENS:
            return None, {
                "reason": "truncated_too_short",
                "schema_tokens": schema_tokens,
                "available_input_tokens": available_input_tokens,
            }

        return {
            "input": query,
            "output": output,
        }, {
            "reason": "kept_truncated",
            "schema_tokens": schema_tokens,
            "available_input_tokens": available_input_tokens,
        }

    return {
        "input": query,
        "output": output,
    }, {
        "reason": "kept",
        "schema_tokens": schema_tokens,
        "available_input_tokens": available_input_tokens,
    }


# Rare harmful labels under-represented in source data; oversample each up to
# --rare-harmful-floor records in train to fix the "0 TP on sexual_content /
# weapons / drugs" problem from v2.
RARE_HARMFUL_LABELS = [
    "sexual_content", "violence", "weapons", "drugs",
    "unauthorized_advice", "self_harm", "misinformation",
    "fraud", "child_exploitation", "malware", "copyright",
]

# Rare adversarial labels — v3 saw indirect_injection recall collapse to 0 because
# the label remap + rare-harmful oversampling pushed down its relative density.
RARE_ADVERSARIAL_LABELS = [
    "indirect_injection", "jailbreak", "prompt_injection",
]


def _safety_ratio(records):
    unsafe = 0
    for r in records:
        for cls in r["output"]["classifications"]:
            if cls["task"] == "safety" and cls["true_label"] == "unsafe":
                unsafe += 1
                break
    return unsafe, len(records) - unsafe


def oversample_train(records, rare_harmful_floor, rare_adversarial_floor,
                     safety_rebalance, rng):
    """Return `records` + extra samples (with replacement) so that:
       - each rare harmful label has >= rare_harmful_floor train occurrences
       - each rare adversarial label has >= rare_adversarial_floor occurrences
       - if safety_rebalance: upsample 'safe' records to restore the
         pre-oversample safe:unsafe ratio (rare-label oversampling is
         unsafe-heavy and otherwise worsens safe recall).

    Test split is never oversampled (called only for train).
    """
    by_harmful = {l: [] for l in RARE_HARMFUL_LABELS}
    by_adversarial = {l: [] for l in RARE_ADVERSARIAL_LABELS}
    for r in records:
        for cls in r["output"]["classifications"]:
            if cls["task"] == "harmful":
                for l in cls["true_label"]:
                    if l in by_harmful:
                        by_harmful[l].append(r)
            elif cls["task"] == "adversarial":
                for l in cls["true_label"]:
                    if l in by_adversarial:
                        by_adversarial[l].append(r)

    added = []
    harm_added = {}
    for label, recs in by_harmful.items():
        if not recs:
            harm_added[label] = 0
            continue
        need = max(0, rare_harmful_floor - len(recs))
        if need > 0:
            added.extend(rng.choices(recs, k=need))
        harm_added[label] = need

    adv_added = {}
    for label, recs in by_adversarial.items():
        if not recs:
            adv_added[label] = 0
            continue
        need = max(0, rare_adversarial_floor - len(recs))
        if need > 0:
            added.extend(rng.choices(recs, k=need))
        adv_added[label] = need

    base_unsafe, base_safe = _safety_ratio(records)
    target_unsafe_ratio = base_unsafe / (base_unsafe + base_safe) if records else 0.0

    expanded = records + added
    new_unsafe, new_safe = _safety_ratio(expanded)
    safe_added = 0
    safe_pool = [r for r in records
                 if any(cls["task"] == "safety" and cls["true_label"] == "safe"
                        for cls in r["output"]["classifications"])]

    if safety_rebalance and safe_pool and new_unsafe + new_safe > 0:
        current_unsafe_ratio = new_unsafe / (new_unsafe + new_safe)
        if current_unsafe_ratio > target_unsafe_ratio:
            total_target = new_unsafe / target_unsafe_ratio
            safe_target = total_target - new_unsafe
            need_safe = int(max(0, safe_target - new_safe))
            if need_safe > 0:
                added.extend(rng.choices(safe_pool, k=need_safe))
                safe_added = need_safe

    final_records = records + added
    final_unsafe, final_safe = _safety_ratio(final_records)

    print(f"\n  Oversampling (train only):")
    print(f"    Rare-harmful floor:     {rare_harmful_floor}")
    for l in RARE_HARMFUL_LABELS:
        base = len(by_harmful[l])
        extra = harm_added[l]
        print(f"      {l:<22} base={base:>5}  +{extra:>5}  -> {base + extra:>5}")
    print(f"    Rare-adversarial floor: {rare_adversarial_floor}")
    for l in RARE_ADVERSARIAL_LABELS:
        base = len(by_adversarial[l])
        extra = adv_added[l]
        print(f"      {l:<22} base={base:>5}  +{extra:>5}  -> {base + extra:>5}")
    print(f"    Safety rebalance:    {'on' if safety_rebalance else 'off'}"
          f"  (safe +{safe_added})")
    print(f"    Safety ratio (unsafe:safe):"
          f" base={base_unsafe}:{base_safe}"
          f" -> final={final_unsafe}:{final_safe}")
    print(f"    Total records added: {len(added)}")
    return final_records


def main():
    parser = argparse.ArgumentParser(
        description="Format all datasets into GLiNER2 guardrail-gliner2 multi-task schema")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with *.train.jsonl and *.test.jsonl")
    parser.add_argument("--output-dir", default="/data",
                        help="Output directory for guardrail_gliner2_{train,test}.jsonl")
    parser.add_argument("--max-per-dataset", type=int, default=0,
                        help="Max records per dataset (0=all)")
    parser.add_argument("--rare-harmful-floor", type=int, default=4000,
                        help="Min train occurrences per rare harmful label (0=disable)")
    parser.add_argument("--rare-adversarial-floor", type=int, default=4000,
                        help="Min train occurrences per rare adversarial label (0=disable). "
                             "Fixes v3's indirect_injection=0 recall collapse.")
    parser.add_argument("--no-safety-rebalance", action="store_true",
                        help="Disable safe-row upsampling that restores the pre-oversample "
                             "safe:unsafe ratio after rare-label oversampling")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for split in ["train", "test"]:
        files = sorted(in_dir.glob(f"*.{split}.jsonl"))
        if not files:
            print(f"No *.{split}.jsonl found in {in_dir}")
            continue

        all_records = []
        stats = Counter()
        safety_stats = Counter()
        adversarial_stats = Counter()
        harmful_stats = Counter()
        formatter_stats = Counter()
        schema_sum = 0
        schema_count = 0

        print(f"\n=== {split.upper()} ===")
        for ds_path in files:
            name = ds_path.stem.replace(f".{split}", "")
            records = []
            with ds_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw = json.loads(line)
                        rec, meta = convert_record(raw)
                        formatter_stats[meta["reason"]] += 1
                        if "schema_tokens" in meta:
                            schema_sum += meta["schema_tokens"]
                            schema_count += 1
                        if rec:
                            records.append(rec)

            # Sample if needed
            if args.max_per_dataset > 0 and len(records) > args.max_per_dataset:
                rng.shuffle(records)
                records = records[:args.max_per_dataset]

            all_records.extend(records)
            stats[name] = len(records)

            # Collect stats
            for rec in records:
                for cls in rec["output"]["classifications"]:
                    if cls["task"] == "safety":
                        safety_stats[cls["true_label"]] += 1
                    elif cls["task"] == "adversarial":
                        for lbl in cls["true_label"]:
                            adversarial_stats[lbl] += 1
                    elif cls["task"] == "harmful":
                        for lbl in cls["true_label"]:
                            harmful_stats[lbl] += 1

            print(f"  {name:<45} {len(records):>7}")

        pre_oversample = len(all_records)
        if split == "train" and (args.rare_harmful_floor > 0 or args.rare_adversarial_floor > 0):
            all_records = oversample_train(
                all_records,
                rare_harmful_floor=args.rare_harmful_floor,
                rare_adversarial_floor=args.rare_adversarial_floor,
                safety_rebalance=not args.no_safety_rebalance,
                rng=rng,
            )
            # Recompute stats after oversampling so the printed summary matches the file.
            safety_stats = Counter()
            adversarial_stats = Counter()
            harmful_stats = Counter()
            for rec in all_records:
                for cls in rec["output"]["classifications"]:
                    if cls["task"] == "safety":
                        safety_stats[cls["true_label"]] += 1
                    elif cls["task"] == "adversarial":
                        for lbl in cls["true_label"]:
                            adversarial_stats[lbl] += 1
                    elif cls["task"] == "harmful":
                        for lbl in cls["true_label"]:
                            harmful_stats[lbl] += 1

        rng.shuffle(all_records)

        out_path = out_dir / f"guardrail_gliner2_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        avg_schema = schema_sum / schema_count if schema_count else 0.0
        print(f"\n  Total: {len(all_records)} (pre-oversample: {pre_oversample})")
        print(f"\n  Formatter report:")
        print(f"    Kept as-is:                  {formatter_stats['kept']:>7}")
        print(f"    Kept after truncation:       {formatter_stats['kept_truncated']:>7}")
        print(f"    Dropped schema-too-large:    {formatter_stats['schema_budget_too_small']:>7}")
        print(f"    Dropped truncated-too-short: {formatter_stats['truncated_too_short']:>7}")
        print(f"    Avg schema tokens:           {avg_schema:>7.1f}")
        print(f"\n  Safety:      {dict(safety_stats.most_common())}")
        print(f"\n  Adversarial: (top 10)")
        for lbl, c in adversarial_stats.most_common(10):
            print(f"    {lbl:<25} {c:>7}")
        print(f"\n  Harmful: (top 10)")
        for lbl, c in harmful_stats.most_common(10):
            print(f"    {lbl:<25} {c:>7}")
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
