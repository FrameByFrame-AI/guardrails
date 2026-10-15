"""Cosine-similarity → calibrated probability.

Cosine similarity is not a probability. Phase 3 fits an isotonic regression on
labelled (cosine, is_block) pairs; the wrapper then computes
``score = calibrated_prob - threshold`` so the API contract's negative-on-PASS
sign convention is preserved.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class IdentityCalibrator:
    def transform(self, sims: np.ndarray) -> np.ndarray:
        return np.clip(sims, 0.0, 1.0)


class IsotonicCalibrator:
    def __init__(self, model) -> None:
        self.model = model

    def transform(self, sims: np.ndarray) -> np.ndarray:
        return self.model.transform(np.asarray(sims, dtype=np.float64))

    @classmethod
    def fit(cls, sims, labels) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(np.asarray(sims, dtype=np.float64), np.asarray(labels, dtype=np.float64))
        return cls(ir)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            return cls(pickle.load(f))


def load_calibrator(path: str | Path | None):
    if path is None:
        return IdentityCalibrator()
    return IsotonicCalibrator.load(path)
