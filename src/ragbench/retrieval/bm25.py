"""BM25 retrieval with transparent tokenization and stable ranking."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ragbench.retrieval.base import Retriever
from ragbench.schemas import Chunk, RetrievalResult


TOKEN_PATTERN = re.compile(r"\b\w+(?:['’-]\w+)*\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        if any(not tokens for tokens in tokenized):
            raise ValueError("BM25 cannot index a chunk with no tokens")
        self._chunks = list(chunks)
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._index is None:
            raise RuntimeError("fit must be called before retrieve")

        scores = self._index.get_scores(tokenize(query))
        # Original corpus order is the deterministic secondary key for ties.
        indices = sorted(range(len(self._chunks)), key=lambda index: (-float(scores[index]), index))
        return [
            RetrievalResult(
                chunk_id=self._chunks[index].chunk_id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(indices[:top_k], start=1)
        ]
