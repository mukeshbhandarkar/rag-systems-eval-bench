from collections.abc import Sequence

import numpy as np
import pytest

from ragbench.retrieval.dense import DenseRetriever
from ragbench.schemas import Chunk, RetrievalResult


class FakeEncoder:
    dimension = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        assert batch_size > 0
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def make_retriever() -> DenseRetriever:
    encoder = FakeEncoder(
        {
            "semantic match": [2.0, 0.0],
            "other": [0.0, 3.0],
            "also other": [0.0, 4.0],
            "query": [1.0, 0.0],
        }
    )
    retriever = DenseRetriever(model_name="fake/test", batch_size=2, encoder=encoder)
    retriever.fit(
        [
            Chunk("match", "d1", "semantic match", 0),
            Chunk("tie-1", "d2", "other", 0),
            Chunk("tie-2", "d3", "also other", 0),
        ]
    )
    return retriever


def test_dense_ranking_contract_top_k_and_normalization() -> None:
    results = make_retriever().retrieve("query", top_k=2)
    assert all(isinstance(result, RetrievalResult) for result in results)
    assert [(result.chunk_id, result.rank) for result in results] == [
        ("match", 1), ("tie-1", 2)
    ]
    assert results[0].score == pytest.approx(1.0)


def test_dense_ties_preserve_corpus_order() -> None:
    results = make_retriever().retrieve("query", top_k=10)
    assert [result.chunk_id for result in results] == ["match", "tie-1", "tie-2"]


def test_dense_validates_configuration_and_lifecycle() -> None:
    encoder = FakeEncoder({"query": [1.0, 0.0]})
    with pytest.raises(ValueError, match="model_name"):
        DenseRetriever(model_name="", encoder=encoder)
    with pytest.raises(ValueError, match="batch_size"):
        DenseRetriever(encoder=encoder, batch_size=0)
    retriever = DenseRetriever(encoder=encoder)
    with pytest.raises(RuntimeError):
        retriever.retrieve("query", 1)
    with pytest.raises(ValueError):
        retriever.fit([])


def test_dense_rejects_zero_embeddings() -> None:
    retriever = DenseRetriever(encoder=FakeEncoder({"empty": [0.0, 0.0]}))
    with pytest.raises(ValueError, match="zero-length"):
        retriever.fit([Chunk("c", "d", "empty", 0)])
