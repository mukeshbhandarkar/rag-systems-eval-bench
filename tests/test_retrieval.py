import pytest

from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.schemas import Chunk


def test_bm25_returns_ranked_results_and_stable_ties() -> None:
    chunks = [
        Chunk("c1", "d1", "python testing tools", 0),
        Chunk("c2", "d2", "gardening soil plants", 0),
        Chunk("c3", "d3", "python package tools", 0),
    ]
    retriever = BM25Retriever()
    retriever.fit(chunks)
    results = retriever.retrieve("gardening plants", top_k=3)
    assert results[0].chunk_id == "c2"
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[1].chunk_id == "c1"  # tied zero scores preserve corpus order


def test_bm25_validates_lifecycle_and_top_k() -> None:
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.retrieve("query", 1)
    with pytest.raises(ValueError):
        retriever.fit([])
