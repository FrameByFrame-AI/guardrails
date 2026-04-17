#!/usr/bin/env python3
"""
Generate adversarial character-obfuscation variants of blocked prompts.

Attack techniques applied (from AccuKnox AI-SPM research + OWASP):
  - Homoglyph substitution      (Latin → Cyrillic/Greek lookalikes)
  - Zero-width char insertion   (U+200B, U+200C, U+200D, U+FEFF)
  - Bidirectional text wrap     (U+202E right-to-left override)
  - Emoji variation smuggling   (U+FE0F appended to chars)
  - Diacritical marks           (combining diacritics)
  - Character spacing           (insert spaces between chars)
  - Unicode tag smuggling       (U+E0000–U+E007F tag block)
  - Upside-down text            (Unicode inverted chars)

For each blocked record, generates `--variants` obfuscated variants. Each variant
preserves the ground truth label — the model should still block it.

Usage:
    python augment_adversarial.py \
        --input /data/processed/train_combined.jsonl \
        --output /data/adversarial_augmented.jsonl \
        --variants 3 \
        --max-records 5000
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Obfuscation techniques

HOMOGLYPHS = {
    # Latin -> Cyrillic/Greek visually identical
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "i": "і", "l": "l",  # Cyrillic/Ukrainian lookalikes
    "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С",
    "H": "Н", "K": "К", "M": "М", "T": "Т", "B": "В",
}

ZERO_WIDTH = ["\u200B", "\u200C", "\u200D", "\uFEFF"]

DIACRITICS = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
    "n": "ñ", "A": "Á", "E": "É", "I": "Í", "O": "Ó",
    "U": "Ú", "N": "Ñ",
}

# Inverted Unicode characters for upside-down text
UPSIDE_DOWN = {
    "a": "ɐ", "b": "q", "c": "ɔ", "d": "p", "e": "ǝ",
    "f": "ɟ", "g": "ƃ", "h": "ɥ", "i": "ᴉ", "j": "ɾ",
    "k": "ʞ", "l": "l", "m": "ɯ", "n": "u", "o": "o",
    "p": "d", "q": "b", "r": "ɹ", "s": "s", "t": "ʇ",
    "u": "n", "v": "ʌ", "w": "ʍ", "x": "x", "y": "ʎ",
    "z": "z",
}


def homoglyph_substitute(text, rate=0.3, rng=None):
    """Replace ~30% of Latin letters with Cyrillic/Greek lookalikes."""
    rng = rng or random
    out = []
    for ch in text:
        if ch in HOMOGLYPHS and rng.random() < rate:
            out.append(HOMOGLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


def insert_zero_width(text, rate=0.3, rng=None):
    """Insert invisible zero-width characters between letters."""
    rng = rng or random
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            out.append(rng.choice(ZERO_WIDTH))
    return "".join(out)


def bidirectional_wrap(text, rng=None):
    """Wrap a portion of text in right-to-left override."""
    rng = rng or random
    if len(text) < 10:
        return text
    # Pick a random slice to reverse
    start = rng.randint(0, len(text) - 6)
    end = rng.randint(start + 3, min(start + 20, len(text)))
    before = text[:start]
    middle = text[start:end][::-1]  # reverse
    after = text[end:]
    return f"{before}\u202E{middle}\u202C{after}"


def emoji_variation_smuggle(text, rate=0.2, rng=None):
    """Append emoji variation selector U+FE0F after characters."""
    rng = rng or random
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalnum() and rng.random() < rate:
            out.append("\uFE0F")
    return "".join(out)


def diacritical_swap(text, rate=0.4, rng=None):
    """Replace vowels with accented forms."""
    rng = rng or random
    out = []
    for ch in text:
        if ch in DIACRITICS and rng.random() < rate:
            out.append(DIACRITICS[ch])
        else:
            out.append(ch)
    return "".join(out)


def char_spacing(text, rng=None):
    """Insert spaces between individual characters in random words."""
    rng = rng or random
    words = text.split()
    if len(words) < 2:
        return text
    n_targets = max(1, len(words) // 4)
    targets = rng.sample(range(len(words)), min(n_targets, len(words)))
    for i in targets:
        if len(words[i]) > 3:
            words[i] = " ".join(words[i])
    return " ".join(words)


def unicode_tag_inject(text, rate=0.1, rng=None):
    """Inject Unicode tag block chars (U+E0020–U+E007E)."""
    rng = rng or random
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < rate:
            # Map to tag range
            tag = chr(0xE0000 + ord(ch.lower()) - ord("a") + 0x20) if "a" <= ch.lower() <= "z" else "\U000E0020"
            out.append(tag)
    return "".join(out)


def upside_down_text(text, rate=0.4, rng=None):
    """Replace letters with upside-down forms."""
    rng = rng or random
    out = []
    for ch in text:
        if ch.lower() in UPSIDE_DOWN and rng.random() < rate:
            replacement = UPSIDE_DOWN[ch.lower()]
            out.append(replacement.upper() if ch.isupper() else replacement)
        else:
            out.append(ch)
    return "".join(out)


TECHNIQUES = [
    ("homoglyph", homoglyph_substitute),
    ("zero_width", insert_zero_width),
    ("bidirectional", bidirectional_wrap),
    ("emoji_smuggle", emoji_variation_smuggle),
    ("diacritical", diacritical_swap),
    ("char_spacing", char_spacing),
    ("unicode_tag", unicode_tag_inject),
    ("upside_down", upside_down_text),
]


# ---------------------------------------------------------------------------
# Main pipeline

def apply_random_technique(query, rng):
    """Pick a technique, apply it, return (variant, technique_name)."""
    name, fn = rng.choice(TECHNIQUES)
    return fn(query, rng=rng), name


def is_attack_record(record):
    """Only augment records that are attack-like (moderation/injection/jailbreak).
    Skip PII records — obfuscation is a different problem there."""
    if not record.get("blocked"):
        return False
    guardrail_type = record.get("type", "")
    if guardrail_type in ("safety-classifier", "rules-based-protections",
                          "output-validation"):
        return True
    if guardrail_type == "moderation":
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="",
                        help="Input JSONL (processed format: {query, blocked, type, answer, topic})")
    parser.add_argument("--input-dir", default="",
                        help="Directory with *.train.jsonl files to scan (alternative to --input)")
    parser.add_argument("--include-datasets", default="",
                        help="Comma-separated list of dataset stems to include from --input-dir "
                             "(e.g. prompt-injections-benchmark,llm-red-teaming-dataset,PIGuard,"
                             "RaccoonBench). Empty = all attack-like records.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=2,
                        help="Obfuscated variants per eligible record")
    parser.add_argument("--max-records", type=int, default=0,
                        help="Max eligible records to augment (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load — either a single file or a directory of *.train.jsonl
    records = []
    if args.input:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {args.input}")
    elif args.input_dir:
        include = set(s.strip() for s in args.include_datasets.split(",") if s.strip())
        in_dir = Path(args.input_dir)
        for path in sorted(in_dir.glob("*.train.jsonl")):
            stem = path.stem.replace(".train", "")
            if include and stem not in include:
                continue
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            print(f"  {stem}: +{sum(1 for _ in open(path))} lines")
        print(f"Loaded {len(records)} total records from {in_dir}")
    else:
        parser.error("Must provide --input or --input-dir")

    eligible = [r for r in records if is_attack_record(r)]
    print(f"Eligible (attack-like blocked): {len(eligible)}")

    if args.max_records > 0 and len(eligible) > args.max_records:
        rng.shuffle(eligible)
        eligible = eligible[:args.max_records]
        print(f"Capped to {args.max_records}")

    # Augment
    augmented = []
    technique_counts = {name: 0 for name, _ in TECHNIQUES}
    for rec in eligible:
        query = rec.get("query", "")
        if not query or len(query) < 10:
            continue
        for _ in range(args.variants):
            variant_query, technique = apply_random_technique(query, rng)
            if variant_query == query:
                continue
            new_rec = dict(rec)
            new_rec["query"] = variant_query
            new_rec["_augmentation"] = technique
            augmented.append(new_rec)
            technique_counts[technique] += 1

    print(f"\nGenerated {len(augmented)} augmented records:")
    for name, count in technique_counts.items():
        print(f"  {name:<20} {count:>6}")

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in augmented:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
