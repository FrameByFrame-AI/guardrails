"""Deterministic train/holdout split shared by prep and eval.

Both ``scripts/prepare_pairs.py`` (training extraction) and
``scripts/prepare_eval_suite.py`` (eval extraction) call ``is_held_out`` with
the same ``(source, row_id, seed)`` so a single physical row never lands on
both sides — no leakage, no need for a separate held-out manifest.
"""

from __future__ import annotations

import hashlib

DEFAULT_SEED = 20260430


def holdout_score(source: str, row_id: str, seed: int = DEFAULT_SEED) -> int:
    """Stable 0..99 hash of ``(source, row_id)``."""
    h = hashlib.sha1(f"{seed}|{source}|{row_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def is_held_out(source: str, row_id: str, holdout_pct: int,
                seed: int = DEFAULT_SEED) -> bool:
    return holdout_score(source, row_id, seed) < holdout_pct
