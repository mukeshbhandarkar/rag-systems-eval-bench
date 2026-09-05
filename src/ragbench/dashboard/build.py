"""Consolidate persisted retrieval runs into dashboard and dataset artifacts."""

from __future__ import annotations

import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

from ragbench.evaluation.retrieval import recall_at_k, reciprocal_rank_at_k
from ragbench.schemas import BenchmarkQuestion, Chunk, Document, to_dict


CUTOFFS = (1, 3, 5)


def build_dashboard_payloads(
    *,
    questions: list[BenchmarkQuestion],
    chunks: list[Chunk],
    runs: dict[str, list[dict[str, Any]]],
    resources: dict[str, Any],
    benchmark_version: str,
    corpus_version: str,
    generation_metrics: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = {"bm25", "dense", "hybrid"}
    if set(runs) != expected:
        raise ValueError(f"Expected retrieval runs {sorted(expected)}")
    question_map = {question.question_id: question for question in questions}
    if len(question_map) != len(questions):
        raise ValueError("Dashboard questions contain duplicate IDs")
    answerable = [question for question in questions if question.gold_chunk_ids]
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    retrievers = []
    records_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("bm25", "dense", "hybrid"):
        records = runs[name]
        by_id = {record["question_id"]: record for record in records}
        if set(by_id) != set(question_map):
            raise ValueError(f"{name} run question IDs do not match the benchmark")
        records_by_system[name] = by_id
        overall = _metrics_for(answerable, by_id)
        category_metrics = {}
        for category in sorted({question.category for question in questions}):
            category_questions = [
                question
                for question in answerable
                if question.category == category
            ]
            category_metrics[category] = {
                "question_count": len(category_questions),
                **(_metrics_for(category_questions, by_id) if category_questions else _empty_metrics()),
            }
        resource = resources.get(name, {})
        retrievers.append(
            {
                "id": name,
                "label": {"bm25": "BM25", "dense": "Dense MiniLM", "hybrid": "Hybrid RRF"}[name],
                "model": "sentence-transformers/all-MiniLM-L6-v2" if name != "bm25" else None,
                "metrics": overall,
                "elapsed_seconds": resource.get("elapsed_seconds"),
                "peak_rss_mb": (
                    round(resource["peak_rss_kb"] / 1024, 1)
                    if resource.get("peak_rss_kb") is not None
                    else None
                ),
                "categories": category_metrics,
            }
        )

    failures = _build_failures(questions, records_by_system, chunk_map)
    category_counts: dict[str, int] = defaultdict(int)
    for question in questions:
        category_counts[question.category] += 1
    results: dict[str, Any] = {
        "benchmark": {
            "name": "RAG Systems Evaluation Bench",
            "benchmark_version": benchmark_version,
            "corpus_version": corpus_version,
            "document_count": len({chunk.doc_id for chunk in chunks}),
            "chunk_count": len(chunks),
            "question_count": len(questions),
            "answerable_count": len(answerable),
            "unanswerable_count": len(questions) - len(answerable),
            "categories": dict(sorted(category_counts.items())),
            "content_origin": "Original synthetic technical passages",
        },
        "retrievers": retrievers,
        "methodology": {
            "cutoffs": list(CUTOFFS),
            "hybrid_rrf_k": 60,
            "hybrid_candidate_k": 20,
            "resource_note": resources.get("measurement_note"),
            "environment": {
                "execution": "CPU-only",
                "python": platform.python_version(),
            },
        },
    }
    if generation_metrics is not None:
        results["generation"] = {
            "retriever": "bm25",
            "model": "google/flan-t5-small",
            "scope": "ragbench-v1 public benchmark",
            "config": generation_config or {},
            "metrics": generation_metrics,
            "elapsed_seconds": resources.get("generation", {}).get("elapsed_seconds"),
            "peak_rss_mb": round(
                resources.get("generation", {}).get("peak_rss_kb", 0) / 1024, 1
            ),
            "semantic_evaluation": {
                "elapsed_seconds": resources.get("answer_semantic_evaluation", {}).get(
                    "elapsed_seconds"
                ),
                "peak_rss_mb": round(
                    resources.get("answer_semantic_evaluation", {}).get("peak_rss_kb", 0)
                    / 1024,
                    1,
                ),
            },
        }
    _assert_finite(results)
    return results, failures


def write_public_packages(
    *,
    results: dict[str, Any],
    failures: list[dict[str, Any]],
    documents: list[Document],
    chunks: list[Chunk],
    questions: list[BenchmarkQuestion],
    runs: dict[str, list[dict[str, Any]]],
    dashboard_data_dir: Path,
    dataset_dir: Path,
    benchmark_results_dir: Path,
) -> None:
    for directory in (dashboard_data_dir, dataset_dir, benchmark_results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(dashboard_data_dir / "results.json", results)
    _write_json(dashboard_data_dir / "failures.json", failures)
    _write_json(benchmark_results_dir / "results.json", results)
    _write_json(benchmark_results_dir / "failures.json", failures)

    _write_jsonl(dataset_dir / "corpus.jsonl", [to_dict(document) for document in documents])
    benchmark_rows = []
    for question in questions:
        row = to_dict(question)
        row["relevant_chunk_ids"] = row.pop("gold_chunk_ids")
        benchmark_rows.append(row)
    _write_jsonl(dataset_dir / "benchmark.jsonl", benchmark_rows)
    _write_jsonl(dataset_dir / "chunks.jsonl", [to_dict(chunk) for chunk in chunks])
    retrieval_rows = [
        {"retriever": retriever, **record}
        for retriever in ("bm25", "dense", "hybrid")
        for record in runs[retriever]
    ]
    _write_jsonl(dataset_dir / "retrieval_results.jsonl", retrieval_rows)
    _write_json(dataset_dir / "benchmark_summary.json", results)
    _write_jsonl(benchmark_results_dir / "retrieval_results.jsonl", retrieval_rows)


def _metrics_for(
    questions: list[BenchmarkQuestion], records: dict[str, dict[str, Any]]
) -> dict[str, float]:
    if not questions:
        raise ValueError("Cannot calculate metrics for an empty question group")
    output: dict[str, float] = {}
    for cutoff in CUTOFFS:
        recalls = []
        reciprocal_ranks = []
        for question in questions:
            ids = [item["chunk_id"] for item in records[question.question_id]["retrieved_results"]]
            recalls.append(recall_at_k(ids, question.gold_chunk_ids, cutoff))
            reciprocal_ranks.append(reciprocal_rank_at_k(ids, question.gold_chunk_ids, cutoff))
        output[f"recall_at_{cutoff}"] = sum(recalls) / len(recalls)
        output[f"mrr_at_{cutoff}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return output


def _empty_metrics() -> dict[str, None]:
    return {
        f"{metric}_at_{cutoff}": None
        for cutoff in CUTOFFS
        for metric in ("recall", "mrr")
    }


def _build_failures(
    questions: list[BenchmarkQuestion],
    systems: dict[str, dict[str, dict[str, Any]]],
    chunks: dict[str, Chunk],
) -> list[dict[str, Any]]:
    failures = []
    for question in questions:
        if not question.gold_chunk_ids:
            continue
        system_rows = {}
        has_failure = False
        for name in ("bm25", "dense", "hybrid"):
            retrieved = systems[name][question.question_id]["retrieved_results"][:5]
            retrieved_ids = [item["chunk_id"] for item in retrieved]
            recall = recall_at_k(retrieved_ids, question.gold_chunk_ids, 5)
            has_failure = has_failure or recall < 1.0
            system_rows[name] = {
                "recall_at_5": recall,
                "results": [
                    {
                        "chunk_id": item["chunk_id"],
                        "rank": item["rank"],
                        "excerpt": chunks[item["chunk_id"]].text[:180],
                    }
                    for item in retrieved[:3]
                ],
            }
        if has_failure:
            failures.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "category": question.category,
                    "relevant_chunk_ids": question.gold_chunk_ids,
                    "reference_answers": question.reference_answers,
                    "retrievers": system_rows,
                }
            )
    return failures


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Dashboard data contains NaN or Infinity")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite(nested)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
