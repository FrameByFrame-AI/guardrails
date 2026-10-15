"""LRU cache for topic embeddings.

Callers reuse topic lists across requests, so encoding `["ransomware", ...]`
once and reusing the matrix is the cheapest available win.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np


class TopicEmbeddingCache:
    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()

    def get_many(
        self,
        topics: list[str],
        template_version: str,
        encode_fn: Callable[[list[str]], np.ndarray],
    ) -> np.ndarray:
        keys = [(template_version, t) for t in topics]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._entries]
        if missing_idx:
            new_vecs = encode_fn([topics[i] for i in missing_idx])
            for j, i in enumerate(missing_idx):
                self._put(keys[i], new_vecs[j])
        return np.stack([self._get(k) for k in keys], axis=0)

    def _get(self, key: tuple[str, str]) -> np.ndarray:
        self._entries.move_to_end(key)
        return self._entries[key]

    def _put(self, key: tuple[str, str], value: np.ndarray) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
