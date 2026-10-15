"""Multilingual sentence splitter.

Naive ``split('.')`` fails on Japanese (``。``), Chinese, and code-switched
text. ``pysbd`` covers ~20 languages rule-based; ``wtpsplit`` is the slower
neural fallback. Phase 2 of the plan picks the winner under the latency budget.
"""

from __future__ import annotations

from functools import lru_cache


def _split_pysbd(text: str, language: str) -> list[str]:
    import pysbd

    seg = pysbd.Segmenter(language=language, clean=False)
    return [s for s in seg.segment(text) if s and s.strip()]


def _split_wtpsplit(text: str, language: str) -> list[str]:
    splitter = _load_wtpsplit()
    return [s for s in splitter.split(text, lang_code=language) if s and s.strip()]


@lru_cache(maxsize=1)
def _load_wtpsplit():
    from wtpsplit import SaT

    return SaT("sat-3l-sm")


def split_sentences(text: str, language: str = "en", backend: str = "pysbd") -> list[str]:
    if not text or not text.strip():
        return []
    if backend == "pysbd":
        try:
            return _split_pysbd(text, language)
        except ValueError:
            return _split_pysbd(text, "en")
    if backend == "wtpsplit":
        return _split_wtpsplit(text, language)
    raise ValueError(f"unknown splitter backend: {backend}")
