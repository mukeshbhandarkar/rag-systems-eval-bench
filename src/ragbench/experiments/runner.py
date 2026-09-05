"""Config-driven retrieval benchmark execution and artifact writing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragbench.evaluation.retrieval import aggregate_metrics, recall_at_k, reciprocal_rank_at_k
from ragbench.generation.base import Generator
from ragbench.retrieval.base import Retriever
from ragbench.schemas import BenchmarkQuestion, Chunk, ExperimentResult, to_dict


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    chunks: list[Chunk],
    questions: list[BenchmarkQuestion],
    retriever: Retriever,
    top_k: int,
    run_id: str,
    corpus_version: str,
    benchmark_version: str,
    output_root: str | Path = "runs",
    retriever_name: str = "bm25",
    retriever_config: dict[str, Any] | None = None,
    generator: Generator | None = None,
    generation_config: dict[str, Any] | None = None,
) -> ExperimentResult:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty filename-safe directory name")
    if not questions:
        raise ValueError("At least one benchmark question is required")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    retriever.fit(chunks)
    timestamp = datetime.now(timezone.utc).isoformat()
    config = {
        "run_id": run_id,
        "retriever": retriever_name,
        "top_k": top_k,
        "corpus_version": corpus_version,
        "benchmark_version": benchmark_version,
        "timestamp": timestamp,
        "retriever_config": retriever_config or {},
    }
    if generator is not None:
        config["generation"] = generation_config or {"model_name": generator.model_name}

    rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    query_metrics: list[dict[str, float]] = []
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for question in questions:
        results = retriever.retrieve(question.question, top_k)
        if question.gold_chunk_ids:
            metrics = {
                "recall_at_k": recall_at_k(results, question.gold_chunk_ids, top_k),
                "reciprocal_rank_at_k": reciprocal_rank_at_k(
                    results, question.gold_chunk_ids, top_k
                ),
            }
            query_metrics.append(metrics)
        else:
            metrics = {"recall_at_k": None, "reciprocal_rank_at_k": None}
        rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "gold_chunk_ids": question.gold_chunk_ids,
                "retrieved_results": [to_dict(result) for result in results],
                **metrics,
            }
        )
        if generator is not None:
            try:
                ranked_chunks = [chunks_by_id[result.chunk_id] for result in results]
            except KeyError as exc:
                raise ValueError(f"Retriever returned unknown chunk ID: {exc.args[0]}") from exc
            generated = generator.generate(question.question, ranked_chunks)
            generation_rows.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "retriever": retriever_name,
                    "retrieved_chunk_ids": [result.chunk_id for result in results],
                    "generator_model": generator.model_name,
                    "generated_answer": generated.answer,
                    "prompt": generated.prompt,
                    "context": generated.context,
                    "context_chunk_ids": generated.context_chunk_ids,
                    "input_token_count": generated.input_token_count,
                    "output_token_count": generated.output_token_count,
                    "truncated": generated.truncated,
                }
            )

    aggregate = aggregate_metrics(query_metrics)
    metrics: dict[str, Any] = {"query_count": len(questions), "top_k": top_k, **aggregate}
    if len(query_metrics) != len(questions):
        metrics["retrieval_evaluated_count"] = len(query_metrics)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    retrieval_path = run_dir / "retrieval.jsonl"
    metrics_path = run_dir / "metrics.json"
    _write_json(config_path, config)
    retrieval_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(metrics_path, metrics)
    generation_path = run_dir / "generation.jsonl"
    if generator is not None:
        generation_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in generation_rows),
            encoding="utf-8",
        )

    output_paths = {
        "config": str(config_path),
        "retrieval": str(retrieval_path),
        "metrics": str(metrics_path),
    }
    if generator is not None:
        output_paths["generation"] = str(generation_path)
    return ExperimentResult(
        run_id=run_id,
        config=config,
        metrics=metrics,
        output_paths=output_paths,
    )
