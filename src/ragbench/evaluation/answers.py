"""Transparent lexical and optional semantic answer metrics."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

import numpy as np

from ragbench.retrieval.dense import TextEncoder


ARTICLES = {"a", "an", "the"}
REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "not enough information",
    "cannot determine from the context",
    "can't determine from the context",
)


def normalize_answer(text: str) -> str:
    """Casefold, replace Unicode punctuation with spaces, remove articles, and collapse space."""
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text.casefold()
    )
    tokens = (token for token in without_punctuation.split() if token not in ARTICLES)
    return " ".join(tokens)


def normalized_exact_match(prediction: str, references: Sequence[str]) -> float:
    if not references:
        raise ValueError("At least one reference answer is required")
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def _token_f1_pair(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    if not references:
        raise ValueError("At least one reference answer is required")
    return max(_token_f1_pair(prediction, reference) for reference in references)


def detected_refusal(prediction: str) -> bool:
    normalized = normalize_answer(prediction)
    return any(normalize_answer(phrase) in normalized for phrase in REFUSAL_PHRASES)


class SemanticAnswerScorer:
    """Maximum normalized embedding cosine similarity over references."""

    def __init__(self, encoder: TextEncoder, *, batch_size: int = 16) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.encoder = encoder
        self.batch_size = batch_size

    def score(self, prediction: str, references: Sequence[str]) -> float:
        if not references:
            raise ValueError("At least one reference answer is required")
        values = np.asarray(
            self.encoder.encode([prediction, *references], batch_size=self.batch_size),
            dtype=np.float32,
        )
        if values.ndim != 2 or values.shape[0] != len(references) + 1:
            raise ValueError("Encoder returned an invalid answer embedding shape")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(values).all():
            raise ValueError("Encoder returned invalid answer embeddings")
        normalized = values / norms
        return float(np.clip(np.max(normalized[1:] @ normalized[0]), -1.0, 1.0))
