import json
from pathlib import Path

import numpy as np
import pytest

from ragbench.evaluation.answers import (
    SemanticAnswerScorer,
    detected_refusal,
    normalize_answer,
    normalized_exact_match,
    token_f1,
)
from ragbench.experiments.answer_runner import (
    aggregate_answer_metrics,
    classify_failure,
    evaluate_answer_records,
    write_answer_artifacts,
)
from ragbench.schemas import BenchmarkQuestion


class FakeAnswerEncoder:
    dimension = 2

    def encode(self, texts, *, batch_size):
        vectors = {
            "prediction": [1.0, 0.0],
            "unrelated": [0.0, 1.0],
            "similar": [2.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_normalization_is_unicode_aware_and_explicit() -> None:
    assert normalize_answer("  THE Python...\tLanguage!  ") == "python language"
    assert normalize_answer("Straße—An Example") == "strasse example"


def test_exact_match_supports_normalization_and_multiple_references() -> None:
    assert normalized_exact_match("Python.", ["python"]) == 1.0
    assert normalized_exact_match("py", ["Python", "Py"]) == 1.0
    assert normalized_exact_match("Python is readable", ["Python"]) == 0.0


def test_token_f1_perfect_partial_none_duplicates_empty_and_references() -> None:
    assert token_f1("Python", ["python."]) == 1.0
    assert token_f1("red blue", ["red green"]) == pytest.approx(0.5)
    assert token_f1("red", ["blue"]) == 0.0
    assert token_f1("red red blue", ["red blue blue"]) == pytest.approx(2 / 3)
    assert token_f1("", [""]) == 1.0
    assert token_f1("", ["nonempty"]) == 0.0
    assert token_f1("automobile", ["car", "automobile"]) == 1.0


def test_semantic_similarity_uses_best_reference() -> None:
    scorer = SemanticAnswerScorer(FakeAnswerEncoder())
    assert scorer.score("prediction", ["unrelated", "similar"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ("I don't know.", True),
        ("There is not enough information in the context.", True),
        ("Python", False),
    ],
)
def test_refusal_detection(prediction: str, expected: bool) -> None:
    assert detected_refusal(prediction) is expected


@pytest.mark.parametrize(
    "retrieval_hit,answer_correct,expected",
    [
        (True, True, "success"),
        (True, False, "generation_failure"),
        (False, False, "retrieval_failure"),
        (False, True, "answer_without_labeled_retrieval"),
    ],
)
def test_failure_decomposition(retrieval_hit: bool, answer_correct: bool, expected: str) -> None:
    assert classify_failure(retrieval_hit=retrieval_hit, answer_correct=answer_correct) == expected


def test_per_question_and_aggregate_metrics() -> None:
    questions = [
        BenchmarkQuestion("success", "q", ["gold"], ["Python"]),
        BenchmarkQuestion("generation", "q", ["gold"], ["BM25"]),
        BenchmarkQuestion("unanswerable", "q", ["none"], [], False),
    ]
    records = [
        {"question_id": "success", "generated_answer": "Python.", "retrieved_chunk_ids": ["gold"]},
        {"question_id": "generation", "generated_answer": "BM25 method", "retrieved_chunk_ids": ["gold"]},
        {"question_id": "unanswerable", "generated_answer": "I do not know", "retrieved_chunk_ids": []},
    ]
    rows, metrics = evaluate_answer_records(questions, records)
    assert [row["failure_type"] for row in rows] == ["success", "generation_failure", None]
    assert metrics["exact_match"] == 0.5
    assert metrics["success_count"] == metrics["generation_failure_count"] == 1
    assert metrics["generation_failure_rate_given_retrieval_hit"] == 0.5
    assert metrics["abstention_accuracy"] == 1.0


def test_aggregate_all_failure_categories() -> None:
    rows = [
        {"answerable": True, "exact_match": float(correct), "token_f1": float(correct),
         "retrieval_hit": hit, "failure_type": classify_failure(retrieval_hit=hit, answer_correct=correct)}
        for hit, correct in [(True, True), (True, False), (False, False), (False, True)]
    ]
    metrics = aggregate_answer_metrics(rows)
    assert metrics["success_count"] == 1
    assert metrics["generation_failure_count"] == 1
    assert metrics["retrieval_failure_count"] == 1
    assert metrics["answer_without_labeled_retrieval_count"] == 1


def test_persisted_generation_evaluation_and_artifact_serialization(tmp_path: Path) -> None:
    questions = [BenchmarkQuestion("q", "question", ["gold"], ["answer"])]
    persisted = tmp_path / "generation.jsonl"
    persisted.write_text(
        json.dumps({"question_id": "q", "generated_answer": "answer", "retrieved_chunk_ids": ["gold"]}) + "\n"
    )
    records = [json.loads(line) for line in persisted.read_text().splitlines()]
    rows, metrics = evaluate_answer_records(questions, records)
    paths = write_answer_artifacts(tmp_path, rows, metrics)
    assert json.loads(Path(paths["answer_metrics"]).read_text())["exact_match"] == 1.0
    assert json.loads(Path(paths["answer_evaluation"]).read_text())["failure_type"] == "success"


def test_mismatched_and_duplicate_ids_fail_cleanly() -> None:
    questions = [BenchmarkQuestion("q", "question", ["gold"], ["answer"])]
    with pytest.raises(ValueError, match="do not match"):
        evaluate_answer_records(
            questions,
            [{"question_id": "other", "generated_answer": "answer", "retrieved_chunk_ids": []}],
        )
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_answer_records(questions + questions, [])
