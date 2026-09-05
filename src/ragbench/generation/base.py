"""Generator contract used by generation-enabled experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragbench.schemas import Chunk, GeneratedAnswer


class Generator(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier recorded in experiment artifacts."""

    @abstractmethod
    def generate(self, question: str, ranked_chunks: list[Chunk]) -> GeneratedAnswer:
        """Generate one answer from chunks in retrieval-rank order."""
