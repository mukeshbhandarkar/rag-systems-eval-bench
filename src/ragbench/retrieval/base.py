"""Retriever contract used by experiment runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragbench.schemas import Chunk, RetrievalResult


class Retriever(ABC):
    @abstractmethod
    def fit(self, chunks: list[Chunk]) -> None:
        """Build retrieval state from chunks."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return at most top_k results in descending relevance order."""
