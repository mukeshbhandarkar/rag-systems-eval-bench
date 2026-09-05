"""Hybrid retrieval through deterministic Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.retrieval.base import Retriever
from ragbench.schemas import Chunk, RetrievalResult


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 20


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]], *, rrf_k: int = DEFAULT_RRF_K
) -> list[RetrievalResult]:
    """Fuse rankings with score sum(1 / (rrf_k + rank))."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for result in ranking:
            if result.rank <= 0:
                raise ValueError("Component result ranks must be greater than zero")
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                rrf_k + result.rank
            )
            best_ranks[result.chunk_id] = min(
                best_ranks.get(result.chunk_id, result.rank), result.rank
            )

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_ranks[chunk_id], chunk_id),
    )
    return [
        RetrievalResult(chunk_id=chunk_id, score=scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever(Retriever):
    """Compose lexical and semantic retrievers using rank-only RRF scores."""

    def __init__(
        self,
        *,
        lexical_retriever: Retriever,
        semantic_retriever: Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        if lexical_retriever is semantic_retriever:
            raise ValueError("Hybrid retrieval requires two distinct retriever instances")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def fit(self, chunks: list[Chunk]) -> None:
        self.lexical_retriever.fit(chunks)
        self.semantic_retriever.fit(chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if top_k > self.candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")
        lexical_results = self.lexical_retriever.retrieve(query, self.candidate_k)
        semantic_results = self.semantic_retriever.retrieve(query, self.candidate_k)
        return reciprocal_rank_fusion(
            [lexical_results, semantic_results], rrf_k=self.rrf_k
        )[:top_k]
