#!/usr/bin/env python3
"""
Generate legitimate-unicode safe examples.

Goal: prevent false positives from adversarial training. When we train the model
to flag homoglyphs, zero-width chars, emoji smuggling, etc. as attacks, we also
need to teach it that these Unicode patterns appear legitimately in normal text:

  - Korean, Japanese, Chinese use non-ASCII scripts
  - Spanish/French/Vietnamese names have diacritics
  - Emojis + flag sequences use zero-width joiners
  - Casual emphasis spacing ("H E L L O")
  - Quoted text with special punctuation
  - Code examples with hyphens/dashes

Output: safe records (blocked=false) that should NOT be flagged.
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Legitimate Unicode templates (safe — blocked=false)

KOREAN_NAMES_LEGIT = [
    "이민호", "김수현", "박서준", "정해인", "손예진", "김지수",
    "이수진", "최민수", "한가인", "송중기", "전지현", "공유",
]

SPANISH_NAMES = [
    "José García", "María Fernández", "Gabriel Núñez", "Álvaro Pérez",
    "Sofía Martínez", "Andrés López", "Lucía González",
]

FRENCH_NAMES = [
    "François Dupont", "Chloé Moreau", "Hélène Lefèvre", "Frédéric Girard",
    "Céline Dubois", "Étienne Lambert",
]

VIETNAMESE_NAMES = [
    "Nguyễn Văn An", "Trần Thị Hạnh", "Lê Thị Mai", "Phạm Quỳnh Anh",
    "Hồ Hoàng Yến",
]

EMOJI_EXAMPLES = [
    "오늘 날씨 좋다 ☀️🌸",
    "Meeting at 3pm 👍",
    "Happy birthday! 🎂🎉",
    "Thanks for the help 🙏",
    "한국 🇰🇷 vs 일본 🇯🇵 축구 경기",
    "Finally weekend! 🎊😊",
    "I love programming 💻❤️",
    "Running late 🏃‍♂️💨",
    "New year resolutions 📅✨",
    "Good luck 🍀🤞",
]

CASUAL_SPACING = [
    "S E R I O U S L Y? that was amazing",
    "H E L L O everyone, nice to meet you",
    "NO W A Y that just happened",
    "W H A T? are you kidding me?",
    "O M G this is awesome",
    "B R E A K I N G news alert",
]

QUOTED_TEXT = [
    'She said "hi" and smiled.',
    'The sign read "No entry".',
    "Don't worry, it's fine.",
    "It's a 'work in progress'.",
    '"Thank you," she whispered.',
    "The file is named 'report.pdf'.",
]

CODE_WITH_HYPHENS = [
    "Press C-x C-s to save in Emacs.",
    "The config flag --no-cache disables caching.",
    "Use `kebab-case` for CSS class names.",
    "Run `npm install` to setup dependencies.",
    "Set `read-only: true` in the config.",
    "The option `--debug-level 3` gives verbose output.",
]

KOREAN_CASUAL = [
    "오늘 점심 뭐 먹을까?",
    "내일 회의 시간 언제죠?",
    "주말에 뭐 하세요?",
    "커피 한 잔 할래요?",
    "최근에 본 영화 추천해주세요",
    "이 문제 해결 방법을 알려주세요",
    "날씨가 많이 추워졌네요",
    "오랜만이에요, 잘 지내셨어요?",
    "프로그래밍 책 추천해주세요",
    "한국어 공부하고 있어요",
]

MULTI_CULTURAL_SENTENCES = [
    # Names with diacritics used in normal greetings
    "{name}씨 반갑습니다.",
    "Nice to meet you, {name}.",
    "Hi {name}, how are you today?",
    "{name}의 프로젝트에 대해 이야기해요.",
    "Let me introduce {name} to the team.",
    "{name} will be joining us tomorrow.",
    "{name}씨는 한국에서 오셨어요.",
]


def build_multi_cultural_records(rng, count):
    """Use template + name combos to generate natural sentences with diacritics."""
    all_names = (KOREAN_NAMES_LEGIT + SPANISH_NAMES + FRENCH_NAMES + VIETNAMESE_NAMES)
    records = []
    for _ in range(count):
        tpl = rng.choice(MULTI_CULTURAL_SENTENCES)
        name = rng.choice(all_names)
        records.append(tpl.format(name=name))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=500,
                        help="Target total records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Combine all legitimate-unicode pools
    queries = []
    queries.extend(EMOJI_EXAMPLES)
    queries.extend(CASUAL_SPACING)
    queries.extend(QUOTED_TEXT)
    queries.extend(CODE_WITH_HYPHENS)
    queries.extend(KOREAN_CASUAL)

    # Multi-cultural name mentions (generate additional)
    name_based = build_multi_cultural_records(rng, args.count - len(queries))
    queries.extend(name_based)

    rng.shuffle(queries)
    queries = queries[:args.count]

    # Shape each as a processed-format safe record
    records = []
    for q in queries:
        records.append({
            "query": q,
            "blocked": False,
            "type": "moderation",   # safe classification through moderation path
            "answer": [],
            "topic": [],
            "_source": "legitimate_unicode",
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} legitimate-unicode safe records")
    print(f"Saved to {out_path}")
    print("\nSample:")
    for r in records[:5]:
        print(f"  {r['query']}")


if __name__ == "__main__":
    main()
