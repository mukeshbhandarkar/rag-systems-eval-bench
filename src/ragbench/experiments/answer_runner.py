"""Evaluate persisted generation records without rerunning retrieval or generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragbench.evaluation.answers import (
    SemanticAnswerScorer,
    detected_refusal,
    normalized_exact_match,
    token_f1,
)
from ragbench.schemas import BenchmarkQuestion


FAILURE_TYPES = (
    "success",
    "generation_failure",
    "retrieval_failure",
    "answer_without_labeled_retrieval",
)


def classify_failure(*, retrieval_hit: bool, answer_correct: bool) -> str:
    if retrieval_hit and answer_correct:
        return "success"
    if retrieval_hit:
        return "generation_failure"
    if answer_correct:
        return "answer_without_labeled_retrieval"
    return "retrieval_failure"


def evaluate_answer_records(
    questions: list[BenchmarkQuestion],
    generation_records: list[dict[str, Any]],
    *,
    semantic_scorer: SemanticAnswerScorer | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_map = _unique_questions(questions)
    generation_map = _unique_generation_records(generation_records)
    if set(question_map) != set(generation_map):
        missing = sorted(set(question_map) - set(generation_map))
        extra = sorted(set(generation_map) - set(question_map))
        raise ValueError(f"Question IDs do not match; missing={missing}, extra={extra}")
    if not any(question.reference_answers for question in questions if question.answerable):
        raise ValueError("No reference answers found for answerable questions")

    rows: list[dict[str, Any]] = []
    for record in generation_records:
        question = question_map[record["question_id"]]
        prediction = record.get("generated_answer")
        retrieved_ids = record.get("retrieved_chunk_ids")
        if not isinstance(prediction, str) or not isinstance(retrieved_ids, list):
            raise ValueError(f"Malformed generation record for {question.question_id}")
        retrieval_hit = bool(set(question.gold_chunk_ids).intersection(retrieved_ids))
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "generated_answer": prediction,
            "reference_answers": question.reference_answers,
            "answerable": question.answerable,
            "retrieval_hit": retrieval_hit,
            "exact_match": None,
            "token_f1": None,
            "failure_type": None,
        }
        if question.answerable and question.reference_answers:
            exact_match = normalized_exact_match(prediction, question.reference_answers)
            row["exact_match"] = exact_match
            row["token_f1"] = token_f1(prediction, question.reference_answers)
            row["answer_correct"] = exact_match == 1.0
            row["failure_type"] = classify_failure(
                retrieval_hit=retrieval_hit, answer_correct=row["answer_correct"]
            )
            if semantic_scorer is not None:
                row["semantic_similarity"] = semantic_scorer.score(
                    prediction, question.reference_answers
                )
        elif not question.answerable:
            row["abstained"] = detected_refusal(prediction)
        rows.append(row)

    return rows, aggregate_answer_metrics(rows, semantic_enabled=semantic_scorer is not None)


def aggregate_answer_metrics(
    rows: list[dict[str, Any]], *, semantic_enabled: bool = False
) -> dict[str, Any]:
    evaluated = [row for row in rows if row["answerable"] and row["exact_match"] is not None]
    if not evaluated:
        raise ValueError("No answerable questions with reference answers were evaluated")
    counts = {failure_type: 0 for failure_type in FAILURE_TYPES}
    for row in evaluated:
        counts[row["failure_type"]] += 1
    retrieval_hits = sum(bool(row["retrieval_hit"]) for row in evaluated)
    metrics: dict[str, Any] = {
        "query_count": len(rows),
        "answerable_evaluated_count": len(evaluated),
        "retrieval_success_rate": retrieval_hits / len(evaluated),
        "exact_match": sum(row["exact_match"] for row in evaluated) / len(evaluated),
        "token_f1": sum(row["token_f1"] for row in evaluated) / len(evaluated),
        "success_count": counts["success"],
        "generation_failure_count": counts["generation_failure"],
        "retrieval_failure_count": counts["retrieval_failure"],
        "answer_without_labeled_retrieval_count": counts[
            "answer_without_labeled_retrieval"
        ],
        "generation_failure_rate_given_retrieval_hit": (
            counts["generation_failure"] / retrieval_hits if retrieval_hits else None
        ),
    }
    if semantic_enabled:
        metrics["semantic_similarity_mean"] = sum(
            row["semantic_similarity"] for row in evaluated
        ) / len(evaluated)
    unanswerable = [row for row in rows if not row["answerable"]]
    if unanswerable:
        metrics["unanswerable_count"] = len(unanswerable)
        metrics["abstention_accuracy"] = sum(row["abstained"] for row in unanswerable) / len(
            unanswerable
        )
    return metrics


def write_answer_artifacts(
    run_dir: str | Path, rows: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, str]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    evaluation_path = directory / "answer_evaluation.jsonl"
    metrics_path = directory / "answer_metrics.json"
    evaluation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"answer_evaluation": str(evaluation_path), "answer_metrics": str(metrics_path)}


def _unique_questions(questions: list[BenchmarkQuestion]) -> dict[str, BenchmarkQuestion]:
    values: dict[str, BenchmarkQuestion] = {}
    for question in questions:
        if question.question_id in values:
            raise ValueError(f"Duplicate benchmark question ID: {question.question_id}")
        values[question.question_id] = question
    return values


def _unique_generation_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Generation record has an invalid question_id")
        if question_id in values:
            raise ValueError(f"Duplicate generation question ID: {question_id}")
        values[question_id] = record
    return values
