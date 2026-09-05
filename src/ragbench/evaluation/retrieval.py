"""Retrieval metrics with explicit multi-label behavior."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.schemas import RetrievalResult


def _validate(gold_chunk_ids: Sequence[str], k: int) -> set[str]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    gold = set(gold_chunk_ids)
    if not gold or "" in gold:
        raise ValueError("gold_chunk_ids must contain at least one non-empty ID")
    return gold


def _ids(results: Sequence[RetrievalResult | str], k: int) -> list[str]:
    return [result if isinstance(result, str) else result.chunk_id for result in results[:k]]


def recall_at_k(
    results: Sequence[RetrievalResult | str], gold_chunk_ids: Sequence[str], k: int
) -> float:
    """Return the fraction of distinct gold chunks present in the first k results."""
    gold = _validate(gold_chunk_ids, k)
    return len(gold.intersection(_ids(results, k))) / len(gold)


def reciprocal_rank_at_k(
    results: Sequence[RetrievalResult | str], gold_chunk_ids: Sequence[str], k: int
) -> float:
    """Return reciprocal rank of the first gold result within k, otherwise zero."""
    gold = _validate(gold_chunk_ids, k)
    for rank, chunk_id in enumerate(_ids(results, k), start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        raise ValueError("Cannot aggregate an empty set of query metrics")
    return {
        "recall_at_k": sum(item["recall_at_k"] for item in per_query) / len(per_query),
        "mrr_at_k": sum(item["reciprocal_rank_at_k"] for item in per_query) / len(per_query),
    }
