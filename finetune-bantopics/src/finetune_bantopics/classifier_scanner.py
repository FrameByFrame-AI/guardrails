"""Classifier-head BanTopics scanner.

Loads a multi-label sigmoid classifier (trained by ``scripts/train_classifier.py``).
Inference: split prompt into sentences → encode all in one batch → sigmoid →
filter to caller-requested topics → max over sentences → compare to threshold.

Single forward pass per scan request (no separate topic encoding). The
``ScanResponse`` shape and the ``score = prob - threshold`` sign convention
match the bi-encoder scanner so the API contract is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from finetune_bantopics.scanner import ScanRequest, ScanResponse  # reuse dataclasses
from finetune_bantopics.splitter import split_sentences
from finetune_bantopics.topics import all_topics


@dataclass
class ClassifierScannerConfig:
    model_name_or_path: str
    splitter_backend: str = "pysbd"
    splitter_language: str = "en"
    device: str | None = None
    encode_batch_size: int = 32
    max_seq_length: int = 128
    attn_implementation: str = "eager"


class ClassifierBanTopicsScanner:
    def __init__(self, config: ClassifierScannerConfig) -> None:
        self.config = config
        self._topic_list = all_topics()
        self._topic_idx = {t: i for i, t in enumerate(self._topic_list)}
        self._tokenizer, self._model, self._device = self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            attn_implementation=self.config.attn_implementation,
        ).to(device)
        model.eval()
        return tokenizer, model, device

    def _normalize_topic(self, t: str) -> str:
        return t.strip().lower().replace("-", " ").replace("_", " ")

    def _resolve_topic_indices(self, topics: list[str]) -> tuple[list[int], list[str]]:
        idxs: list[int] = []
        missing: list[str] = []
        for t in topics:
            i = self._topic_idx.get(self._normalize_topic(t))
            if i is None:
                missing.append(t)
            else:
                idxs.append(i)
        return idxs, missing

    def scan(self, request: ScanRequest) -> ScanResponse:
        import torch

        params = request.scanner_params or {}
        topics: list[str] = list(params.get("topics") or [])
        threshold: float = float(params.get("threshold", 0.5))

        if not topics or not request.prompt or not request.prompt.strip():
            return ScanResponse(True, request.scanner_name, -threshold)

        idxs, _missing = self._resolve_topic_indices(topics)
        if not idxs:
            # None of the requested topics are in the trained label set.
            return ScanResponse(True, request.scanner_name, -threshold)

        sentences = split_sentences(
            request.prompt,
            language=self.config.splitter_language,
            backend=self.config.splitter_backend,
        ) or [request.prompt]

        with torch.no_grad():
            enc = self._tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            ).to(self._device)
            logits = self._model(**enc).logits          # S × n_labels
            probs = torch.sigmoid(logits)               # S × n_labels
            score = float(probs[:, idxs].max().item())  # max over sentences × topics

        return ScanResponse(
            is_valid=score < threshold,
            scanner_name=request.scanner_name,
            score=score - threshold,
        )
