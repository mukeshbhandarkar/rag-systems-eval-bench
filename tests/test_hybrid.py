import json
from pathlib import Path

import pytest

from ragbench.experiments.runner import run_experiment
from ragbench.retrieval.base import Retriever
from ragbench.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from ragbench.schemas import BenchmarkQuestion, Chunk, RetrievalResult


def ranking(*chunk_ids: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk_id=chunk_id, score=99.0 - rank, rank=rank)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


class RecordingRetriever(Retriever):
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.fit_calls = 0
        self.requested_k: list[int] = []

    def fit(self, chunks: list[Chunk]) -> None:
        self.fit_calls += 1

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.requested_k.append(top_k)
        return self.results[:top_k]


def test_rrf_exact_math_and_shared_candidate_advantage() -> None:
    fused = reciprocal_rank_fusion(
        [ranking("lexical", "shared", "semantic"), ranking("shared", "semantic", "lexical")],
        rrf_k=60,
    )
    assert [result.chunk_id for result in fused] == ["shared", "lexical", "semantic"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 63)


def test_lexical_and_semantic_only_candidates_are_retained() -> None:
    fused = reciprocal_rank_fusion(
        [ranking("lexical-only", "shared"), ranking("semantic-only", "shared")], rrf_k=60
    )
    assert {result.chunk_id for result in fused} == {
        "lexical-only", "semantic-only", "shared"
    }
    assert fused[0].chunk_id == "shared"


def test_hybrid_uses_candidate_depth_and_returns_top_k_contract() -> None:
    lexical = RecordingRetriever(ranking("a", "b", "c", "d"))
    semantic = RecordingRetriever(ranking("d", "c", "b", "a"))
    hybrid = HybridRetriever(
        lexical_retriever=lexical,
        semantic_retriever=semantic,
        rrf_k=60,
        candidate_k=4,
    )
    hybrid.fit([Chunk("a", "doc", "text", 0)])
    results = hybrid.retrieve("query", top_k=2)
    assert lexical.fit_calls == semantic.fit_calls == 1
    assert lexical.requested_k == semantic.requested_k == [4]
    assert len(results) == 2
    assert all(isinstance(result, RetrievalResult) for result in results)
    assert [result.rank for result in results] == [1, 2]


def test_rrf_ties_use_best_rank_then_chunk_id() -> None:
    best_rank_break = reciprocal_rank_fusion([ranking("z", "x"), ranking("a")], rrf_k=0)
    assert [result.chunk_id for result in best_rank_break] == ["a", "z", "x"]
    identity_break = reciprocal_rank_fusion([ranking("z"), ranking("a")], rrf_k=60)
    assert [result.chunk_id for result in identity_break] == ["a", "z"]


def test_hybrid_validates_configuration() -> None:
    first = RecordingRetriever([])
    second = RecordingRetriever([])
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(lexical_retriever=first, semantic_retriever=second, candidate_k=0)
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(lexical_retriever=first, semantic_retriever=second, rrf_k=-1)
    hybrid = HybridRetriever(
        lexical_retriever=first, semantic_retriever=second, candidate_k=2
    )
    with pytest.raises(ValueError, match="candidate_k"):
        hybrid.retrieve("query", top_k=3)


def test_runner_accepts_hybrid_and_writes_compatible_artifacts(tmp_path: Path) -> None:
    hybrid = HybridRetriever(
        lexical_retriever=RecordingRetriever(ranking("gold")),
        semantic_retriever=RecordingRetriever(ranking("gold")),
        candidate_k=2,
    )
    result = run_experiment(
        chunks=[Chunk("gold", "doc", "text", 0)],
        questions=[BenchmarkQuestion("q", "query", ["gold"])],
        retriever=hybrid,
        retriever_name="hybrid",
        retriever_config={"fusion_method": "reciprocal_rank_fusion", "rrf_k": 60},
        top_k=1,
        run_id="hybrid-run",
        corpus_version="test-corpus",
        benchmark_version="test-benchmark",
        output_root=tmp_path,
    )
    row = json.loads((tmp_path / "hybrid-run" / "retrieval.jsonl").read_text())
    assert result.config["retriever"] == "hybrid"
    assert result.metrics["mrr_at_k"] == 1.0
    assert set(row["retrieved_results"][0]) == {"chunk_id", "score", "rank"}
