#!/usr/bin/env python3
"""Verify exactly how the ko-ettin tokenizer splits Korean dates + suffixes."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/models/outputs/ko-ettin-encoder-150m-ko-pii")
added = set(tok.get_added_vocab().keys())  # our +16k surface tokens (+specials)
print(f"added_vocab size: {len(added)}\n")

cases = [
    "90년 9월 18일생이거든요.",
    "1996년 12월 31일에 태어난",
    "2020년 8월 1일부터 지금",
    "92년 10월 12일이에요.",
    "1992년 3월 6일입니다.",   # the 'good' case
    "1962년 3월 3일 쓰여있고",  # followed by space
]
for text in cases:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    print(f"TEXT: {text!r}")
    for t, (s, e) in zip(toks, enc["offset_mapping"]):
        surf = text[s:e]
        tag = "ADDED" if t in added else "base"
        print(f"   {t!r:>12}  off=({s},{e})  surface={surf!r:>8}  [{tag}]")
    print()

# Direct question: is '일생' / '일에' a single token, and is it ours?
for probe in ["일생", "일에", "일부터", "일이", "일", "일생이", "18일생"]:
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"tokenize({probe!r}) -> {pieces}  (n={len(ids)})  "
          f"added_hit={[p for p in pieces if p in added]}")
