"""Shared span-boundary normalization for benchmark_pii_heldout.py and analyze_errors.py.

The model consistently emits a leading/trailing whitespace (or adjacent
punctuation) character on some spans (documented in README.md's Known
Limitations). Both scripts must trim the same way before comparing spans —
otherwise one script's "boundary error" is the other script's "exact match"
for the same prediction, which is what happened before this module existed.
"""

from __future__ import annotations

import unicodedata


def is_ignorable_boundary_char(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def trim_span_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    trimmed_start = start
    trimmed_end = end

    while trimmed_start < trimmed_end and is_ignorable_boundary_char(text[trimmed_start]):
        trimmed_start += 1
    while trimmed_end > trimmed_start and is_ignorable_boundary_char(text[trimmed_end - 1]):
        trimmed_end -= 1

    if trimmed_start >= trimmed_end:
        return start, end
    return trimmed_start, trimmed_end


def normalize_span_set(
    text: str, spans: set[tuple[str, int, int]]
) -> set[tuple[str, int, int]]:
    normalized: set[tuple[str, int, int]] = set()
    for label, start, end in spans:
        norm_start, norm_end = trim_span_boundaries(text, start, end)
        normalized.add((label, norm_start, norm_end))
    return normalized
