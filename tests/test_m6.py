import json
import math
from pathlib import Path

import pytest

from ragbench.benchmark.validation import validate_public_benchmark
from ragbench.dashboard.build import build_dashboard_payloads
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import BenchmarkQuestion, Chunk


ROOT = Path(__file__).parents[1]


def jsonl(path: Path, record_type=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record_type(**row) for row in rows] if record_type else rows


def public_inputs():
    documents = load_documents(ROOT / "data/public/raw")
    chunks = jsonl(ROOT / "data/public/processed/chunks.jsonl", Chunk)
    questions = jsonl(ROOT / "data/public/benchmark/questions.jsonl", BenchmarkQuestion)
    return documents, chunks, questions


def test_public_benchmark_validation_and_references() -> None:
    documents, chunks, questions = public_inputs()
    summary = validate_public_benchmark(
        documents=documents,
        chunks=chunks,
        questions=questions,
        corpus_metadata=json.loads(
            (ROOT / "data/public/processed/corpus_metadata.json").read_text()
        ),
        benchmark_metadata=json.loads(
            (ROOT / "data/public/benchmark/benchmark_metadata.json").read_text()
        ),
    )
    assert summary["document_count"] == 12
    assert summary["chunk_count"] == 36
    assert summary["question_count"] == 24
    assert summary["answerable_count"] == 22
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    assert all(set(question.gold_chunk_ids) <= chunk_ids for question in questions)


def test_validator_rejects_bad_category_and_duplicate_id() -> None:
    documents, chunks, questions = public_inputs()
    questions[0] = BenchmarkQuestion(
        question_id=questions[1].question_id,
        question=questions[0].question,
        gold_chunk_ids=questions[0].gold_chunk_ids,
        reference_answers=questions[0].reference_answers,
        category="not-valid",
    )
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        validate_public_benchmark(
            documents=documents,
            chunks=chunks,
            questions=questions,
            corpus_metadata={"corpus_version": "v", "document_count": 12, "chunk_count": 36},
            benchmark_metadata={"corpus_version": "v", "benchmark_version": "v", "question_count": 24},
        )


def test_dashboard_summary_mapping_ranges_and_failures() -> None:
    _, chunks, questions = public_inputs()
    persisted = jsonl(ROOT / "benchmark_results/ragbench-v1/retrieval_results.jsonl")
    runs = {
        name: [{key: value for key, value in row.items() if key != "retriever"}
               for row in persisted if row["retriever"] == name]
        for name in ("bm25", "dense", "hybrid")
    }
    resources = json.loads((ROOT / "configs/m6_resources.json").read_text())
    results, failures = build_dashboard_payloads(
        questions=questions,
        chunks=chunks,
        runs=runs,
        resources=resources,
        benchmark_version="ragbench-v1",
        corpus_version="ragbench-v1",
    )
    assert [item["id"] for item in results["retrievers"]] == ["bm25", "dense", "hybrid"]
    assert results["retrievers"][0]["metrics"]["recall_at_5"] == pytest.approx(21 / 22)
    assert results["retrievers"][1]["metrics"]["recall_at_3"] == 1.0
    assert failures[0]["question_id"] == "ragbench-q006"
    assert failures[0]["retrievers"]["bm25"]["recall_at_5"] == 0.0


def test_checked_in_dashboard_data_is_finite_and_consistent() -> None:
    results = json.loads((ROOT / "hf_space/data/results.json").read_text())
    failures = json.loads((ROOT / "hf_space/data/failures.json").read_text())
    assert results["benchmark"]["question_count"] == 24
    assert results["benchmark"]["chunk_count"] == 36
    assert {item["id"] for item in results["retrievers"]} == {"bm25", "dense", "hybrid"}
    for retriever in results["retrievers"]:
        for value in retriever["metrics"].values():
            assert math.isfinite(value) and 0 <= value <= 1
    question_ids = {
        question.question_id
        for question in jsonl(
            ROOT / "data/public/benchmark/questions.jsonl", BenchmarkQuestion
        )
    }
    chunk_ids = {
        chunk.chunk_id
        for chunk in jsonl(ROOT / "data/public/processed/chunks.jsonl", Chunk)
    }
    assert all(item["question_id"] in question_ids for item in failures)
    assert all(set(item["relevant_chunk_ids"]) <= chunk_ids for item in failures)
    assert all(
        (ROOT / "hf_space" / filename).is_file()
        for filename in ("index.html", "styles.css", "app.js", "README.md")
    )


def test_old_fixture_schema_defaults_remain_usable() -> None:
    question = BenchmarkQuestion("fixture", "question", ["chunk"])
    assert question.reference_answers == []
    assert question.answerable is True
    assert question.category == ""
