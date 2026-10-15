"""BanTopics scanner.

Sentence-level bi-encoder zero-shot classifier. Given a request with arbitrary
caller-supplied topics, encode prompt sentences and topics with a shared
multilingual encoder (default: ``ibm-granite/granite-embedding-97m-multilingual-r2``),
take per-topic max similarity over sentences, calibrate, then compare against
the caller-supplied threshold.

Score sign convention (must match existing wrapper):
    score = calibrated_prob - threshold
    is_valid = score < 0          # negative => PASS, positive => BLOCK
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from finetune_bantopics.calibrator import IdentityCalibrator, load_calibrator
from finetune_bantopics.splitter import split_sentences
from finetune_bantopics.topic_cache import TopicEmbeddingCache


DEFAULT_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
DEFAULT_TEMPLATE = "This text is about {topic}."
TEMPLATE_VERSION = "v1"


@dataclass
class ScanRequest:
    scanner_name: str
    scanner_type: str
    scanner_params: dict[str, Any]
    prompt: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanRequest":
        return cls(
            scanner_name=payload["scanner_name"],
            scanner_type=payload["scanner_type"],
            scanner_params=payload.get("scanner_params") or {},
            prompt=payload["prompt"],
        )


@dataclass
class ScanResponse:
    is_valid: bool
    scanner_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "scanner_name": self.scanner_name,
            "score": round(float(self.score), 4),
        }


@dataclass
class ScannerConfig:
    model_name_or_path: str = DEFAULT_MODEL
    topic_template: str = DEFAULT_TEMPLATE
    template_version: str = TEMPLATE_VERSION
    splitter_backend: str = "pysbd"
    splitter_language: str = "en"
    pooling: str = "max"           # {"max", "top3-mean"}
    device: str | None = None
    calibrator_path: str | Path | None = None
    topic_cache_size: int = 4096
    encode_batch_size: int = 32


class BanTopicsScanner:
    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()
        self._encoder = self._load_encoder()
        self._topic_cache = TopicEmbeddingCache(max_entries=self.config.topic_cache_size)
        self._calibrator = load_calibrator(self.config.calibrator_path)

    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        device = self.config.device or _autodetect_device()
        return SentenceTransformer(self.config.model_name_or_path, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = int(self._encoder.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)
        vecs = self._encoder.encode(
            texts,
            batch_size=self.config.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)

    def _encode_topics(self, topics: list[str]) -> np.ndarray:
        return self._topic_cache.get_many(
            topics=topics,
            template_version=self.config.template_version,
            encode_fn=lambda batch: self.encode(
                [self.config.topic_template.format(topic=t) for t in batch]
            ),
        )

    def scan(self, request: ScanRequest) -> ScanResponse:
        params = request.scanner_params or {}
        topics: list[str] = list(params.get("topics") or [])
        threshold: float = float(params.get("threshold", 0.5))

        if not topics or not request.prompt or not request.prompt.strip():
            return ScanResponse(
                is_valid=True,
                scanner_name=request.scanner_name,
                score=-threshold,
            )

        sentences = split_sentences(
            request.prompt,
            language=self.config.splitter_language,
            backend=self.config.splitter_backend,
        )
        if not sentences:
            sentences = [request.prompt]

        sent_vecs = self.encode(sentences)               # S x D
        topic_vecs = self._encode_topics(topics)         # T x D
        sims = sent_vecs @ topic_vecs.T                  # S x T (vectors are normalised)

        per_topic_doc = _pool(sims, mode=self.config.pooling)   # T
        max_sim = float(per_topic_doc.max())
        prob = float(self._calibrator.transform(np.array([max_sim]))[0])
        score = prob - threshold

        return ScanResponse(
            is_valid=score < 0,
            scanner_name=request.scanner_name,
            score=score,
        )


def _pool(sims: np.ndarray, mode: str) -> np.ndarray:
    if sims.size == 0:
        return np.zeros(sims.shape[1] if sims.ndim == 2 else 0, dtype=np.float32)
    if mode == "max":
        return sims.max(axis=0)
    if mode == "top3-mean":
        k = min(3, sims.shape[0])
        topk = np.partition(sims, -k, axis=0)[-k:]
        return topk.mean(axis=0)
    raise ValueError(f"unknown pooling mode: {mode}")


def _autodetect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def scan_batch(
    scanner: BanTopicsScanner, requests: Iterable[ScanRequest]
) -> list[ScanResponse]:
    return [scanner.scan(r) for r in requests]
