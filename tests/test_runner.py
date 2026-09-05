import json
from pathlib import Path

from ragbench.experiments.runner import run_experiment
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.dense import DenseRetriever
from ragbench.schemas import BenchmarkQuestion, Chunk


class RunnerFakeEncoder:
    dimension = 2

    def encode(self, texts, *, batch_size):
        return __import__("numpy").asarray(
            [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]
        )


def test_experiment_output_schema(tmp_path: Path) -> None:
    result = run_experiment(
        chunks=[Chunk("c1", "d1", "alpha beta gamma", 0)],
        questions=[BenchmarkQuestion("q1", "alpha", ["c1"])],
        retriever=BM25Retriever(),
        top_k=5,
        run_id="test-run",
        corpus_version="test-corpus",
        benchmark_version="test-benchmark",
        output_root=tmp_path,
    )
    run_dir = tmp_path / "test-run"
    assert set(path.name for path in run_dir.iterdir()) == {
        "config.json", "retrieval.jsonl", "metrics.json"
    }
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics == {"query_count": 1, "recall_at_k": 1.0, "mrr_at_k": 1.0, "top_k": 5}
    row = json.loads((run_dir / "retrieval.jsonl").read_text())
    assert set(row) == {
        "question_id", "question", "gold_chunk_ids", "retrieved_results",
        "recall_at_k", "reciprocal_rank_at_k"
    }
    assert result.output_paths["metrics"] == str(run_dir / "metrics.json")


def test_experiment_runner_accepts_dense_retriever(tmp_path: Path) -> None:
    result = run_experiment(
        chunks=[Chunk("c1", "d1", "alpha", 0), Chunk("c2", "d2", "beta", 0)],
        questions=[BenchmarkQuestion("q1", "alpha question", ["c1"])],
        retriever=DenseRetriever(model_name="fake", encoder=RunnerFakeEncoder()),
        retriever_name="dense",
        retriever_config={"model_name": "fake", "embedding_dimension": 2},
        top_k=1,
        run_id="dense-run",
        corpus_version="test-corpus",
        benchmark_version="test-benchmark",
        output_root=tmp_path,
    )
    assert result.metrics["recall_at_k"] == 1.0
    assert result.config["retriever"] == "dense"
    assert result.config["retriever_config"]["embedding_dimension"] == 2
