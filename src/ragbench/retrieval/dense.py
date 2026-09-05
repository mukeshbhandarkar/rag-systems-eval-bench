"""CPU-only dense retrieval using normalized embeddings and NumPy."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from ragbench.retrieval.base import Retriever
from ragbench.schemas import Chunk, RetrievalResult


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    """Small injection point that keeps model loading out of ranking logic."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.floating]: ...


class SentenceTransformerEncoder:
    """Thin CPU-only adapter around Sentence Transformers."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError("Install project dependencies with `uv sync`") from exc

        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            raise ValueError(f"Could not load embedding model {model_name!r}: {exc}") from exc
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:  # Sentence Transformers before the method rename.
            dimension_getter = self._model.get_sentence_embedding_dimension
        dimension = dimension_getter()
        if not dimension:
            raise ValueError(f"Embedding model {model_name!r} did not report a dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> NDArray[np.floating]:
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )


def _normalized_rows(values: NDArray[np.floating], *, expected_rows: int) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ValueError(
            f"Encoder returned shape {matrix.shape}; expected ({expected_rows}, embedding_dimension)"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Encoder returned non-finite embedding values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder returned a zero-length embedding")
    return matrix / norms


class DenseRetriever(Retriever):
    """Rank chunks by cosine similarity between normalized embedding vectors."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 16,
        encoder: TextEncoder | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = encoder if encoder is not None else SentenceTransformerEncoder(model_name)
        self._chunks: list[Chunk] = []
        self._embeddings: NDArray[np.float32] | None = None

    @property
    def embedding_dimension(self) -> int:
        return int(self._encoder.dimension)

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Dense retrieval requires at least one chunk")
        embeddings = self._encoder.encode(
            [chunk.text for chunk in chunks], batch_size=self.batch_size
        )
        normalized = _normalized_rows(embeddings, expected_rows=len(chunks))
        if normalized.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Encoder reported dimension {self.embedding_dimension} but returned "
                f"dimension {normalized.shape[1]}"
            )
        self._chunks = list(chunks)
        self._embeddings = normalized

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._embeddings is None:
            raise RuntimeError("fit must be called before retrieve")
        query_embedding = _normalized_rows(
            self._encoder.encode([query], batch_size=self.batch_size), expected_rows=1
        )
        if query_embedding.shape[1] != self._embeddings.shape[1]:
            raise ValueError("Query and corpus embedding dimensions do not match")

        scores = self._embeddings @ query_embedding[0]
        # lexsort uses corpus index as the deterministic secondary key for ties.
        indices = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
        return [
            RetrievalResult(
                chunk_id=self._chunks[int(index)].chunk_id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        ]
