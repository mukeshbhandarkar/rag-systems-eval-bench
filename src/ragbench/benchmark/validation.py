"""Validation rules for publishable benchmark inputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ragbench.schemas import BenchmarkQuestion, Chunk, Document


VALID_CATEGORIES = {
    "lexical",
    "semantic",
    "acronym",
    "distractor",
    "paraphrase",
    "control",
    "unanswerable",
}


def validate_public_benchmark(
    *,
    documents: list[Document],
    chunks: list[Chunk],
    questions: list[BenchmarkQuestion],
    corpus_metadata: dict[str, Any],
    benchmark_metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    _check_unique("document ID", [document.doc_id for document in documents], errors)
    _check_unique("chunk ID", [chunk.chunk_id for chunk in chunks], errors)
    _check_unique("question ID", [question.question_id for question in questions], errors)
    _check_unique(
        "exact question text", [question.question.strip() for question in questions], errors
    )

    document_ids = {document.doc_id for document in documents}
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    for chunk in chunks:
        if chunk.doc_id not in document_ids:
            errors.append(f"Chunk {chunk.chunk_id} references unknown document {chunk.doc_id}")
    for question in questions:
        if not question.question.strip():
            errors.append(f"Question {question.question_id} has empty text")
        if question.category not in VALID_CATEGORIES:
            errors.append(
                f"Question {question.question_id} has invalid category {question.category!r}"
            )
        missing = sorted(set(question.gold_chunk_ids) - chunk_ids)
        if missing:
            errors.append(f"Question {question.question_id} has unknown relevant chunks: {missing}")
        if any(not answer.strip() for answer in question.reference_answers):
            errors.append(f"Question {question.question_id} has an empty reference answer")
        if question.answerable:
            if not question.reference_answers:
                errors.append(f"Answerable question {question.question_id} has no references")
            if not question.gold_chunk_ids:
                errors.append(f"Answerable question {question.question_id} has no relevant chunks")
            if question.category == "unanswerable":
                errors.append(f"Answerable question {question.question_id} uses unanswerable category")
        else:
            if question.gold_chunk_ids or question.reference_answers:
                errors.append(
                    f"Unanswerable question {question.question_id} must not carry gold evidence or answers"
                )
            if question.category != "unanswerable":
                errors.append(
                    f"Unanswerable question {question.question_id} must use unanswerable category"
                )

    corpus_version = corpus_metadata.get("corpus_version")
    if corpus_version != benchmark_metadata.get("corpus_version"):
        errors.append("Corpus versions do not agree between metadata files")
    if benchmark_metadata.get("benchmark_version") != corpus_version:
        errors.append("Public corpus and benchmark must use the coordinated ragbench version")
    if corpus_metadata.get("document_count") != len(documents):
        errors.append("Corpus metadata document_count does not match loaded documents")
    if corpus_metadata.get("chunk_count") != len(chunks):
        errors.append("Corpus metadata chunk_count does not match chunks")
    if benchmark_metadata.get("question_count") != len(questions):
        errors.append("Benchmark metadata question_count does not match questions")

    if errors:
        raise ValueError("Benchmark validation failed:\n- " + "\n- ".join(errors))
    categories = Counter(question.category for question in questions)
    return {
        "corpus_version": corpus_version,
        "benchmark_version": benchmark_metadata["benchmark_version"],
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "answerable_count": sum(question.answerable for question in questions),
        "unanswerable_count": sum(not question.answerable for question in questions),
        "categories": dict(sorted(categories.items())),
    }


def _check_unique(label: str, values: list[str], errors: list[str]) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate {label}s: {duplicates}")
