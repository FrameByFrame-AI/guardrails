"""Multi-label dataset wrapper for the Accuknox CodeLanguage parquet.

The Accuknox dataset (`Accuknoxtechnologies/CodeLanguage`) has three columns:

    prompt   str           — user message, optionally containing code
    target   str (JSON)    — {"is_valid": bool, "category": {"<Lang>": true, ...}}
    kind     str           — "single" | "multi" | "benign"

The target schema is *exactly* the output format we want the model to produce.
This loader returns the prompt text plus a 25-dim multi-hot label vector
indexed by `LANGUAGES` below. Benign rows produce an all-zeros label vector.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Unified 113-language taxonomy (107 lite canonical + 6 config formats). The
# canonical names, ordering, and alias resolution live in `taxonomy` so every
# data-prep script and the trainer share one source of truth.
from code_language_id.taxonomy import (  # noqa: E402
    LANG_TO_IDX,
    LANGUAGES,
    NUM_LANGUAGES,
    resolve,
)


def parse_target(raw: str | dict) -> tuple[bool, np.ndarray]:
    """Parse a `target` cell into (is_valid, multi_hot[NUM_LANGUAGES])."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    is_valid = bool(obj.get("is_valid", False))
    multi_hot = np.zeros(NUM_LANGUAGES, dtype=np.float32)
    for lang in (obj.get("category") or {}):
        # resolve() maps any source spelling onto a canonical label and raises
        # on an unknown one, so a mislabel can never be silently dropped.
        multi_hot[LANG_TO_IDX[resolve(lang)]] = 1.0
    return is_valid, multi_hot


def _stable_rng(seed: int, epoch: int, index: int) -> random.Random:
    payload = f"{seed}:{epoch}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def _variable_window(text: str, min_chars: int, max_chars: int, rng: random.Random) -> str:
    """Return a random contiguous slice in [min_chars, max_chars] characters.

    Mirrors the 100+ trainer's variable_window — at every epoch each row is
    re-sampled to a different length and offset, giving the model multiple
    distinct training views of the same source text.
    """
    if len(text) <= min_chars:
        return text
    upper = min(max_chars, len(text))
    lower = min(min_chars, upper)
    target = rng.randint(lower, upper)
    if target >= len(text):
        return text
    start = rng.randint(0, len(text) - target)
    return text[start : start + target]


class CodeLanguageMultilabelDataset:
    """Read Accuknox parquet and expose (text, multi_hot, is_valid) rows.

    With ``variable_window=True`` (training only), single-language and benign
    rows are resampled to a random byte window each epoch via the stable RNG;
    multi-language rows are returned unmodified — a random window through a
    polyglot prompt would routinely drop one of the labeled languages and
    teach the wrong supervision.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        *,
        variable_window: bool = False,
        min_chars: int = 256,
        max_chars: int = 4096,
        seed: int = 20260601,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.frame = pd.read_parquet(self.parquet_path)
        required = {"prompt", "target"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(
                f"{self.parquet_path} is missing required columns: {sorted(missing)}"
            )
        parsed = [parse_target(t) for t in self.frame["target"]]
        self._is_valid = np.array([p[0] for p in parsed], dtype=bool)
        self._labels = np.stack([p[1] for p in parsed], axis=0)
        self._kinds = (
            self.frame["kind"].astype(str).to_numpy()
            if "kind" in self.frame.columns
            else np.array(["unknown"] * len(self.frame))
        )

        self.variable_window = variable_window
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Bump the epoch so variable_window resamples differently next pass."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        text = row["prompt"]
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        kind = self._kinds[index]
        if (
            self.variable_window
            and kind in ("single", "benign")
            and len(text) > self.min_chars
        ):
            rng = _stable_rng(self.seed, self.epoch, index)
            text = _variable_window(text, self.min_chars, self.max_chars, rng)

        return {
            "text": text,
            "labels": self._labels[index],
            "is_valid": bool(self._is_valid[index]),
            "kind": kind,
        }
