#!/usr/bin/env python3
"""Evaluate persisted generated answers with lexical and optional semantic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeVar

from ragbench.evaluation.answers import SemanticAnswerScorer
from ragbench.experiments.answer_runner import evaluate_answer_records, write_answer_artifacts
from ragbench.schemas import BenchmarkQuestion


T = TypeVar("T")


def load_jsonl(path: Path, record_type: type[T] | None = None) -> list[T] | list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Required JSONL file does not exist: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(record_type(**value) if record_type else value)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid record in {path} at line {line_number}: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument(
        "--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_jsonl(args.questions, BenchmarkQuestion)
    generations = load_jsonl(args.run_dir / "generation.jsonl")
    scorer = None
    if args.semantic:
        from ragbench.retrieval.dense import SentenceTransformerEncoder

        scorer = SemanticAnswerScorer(
            SentenceTransformerEncoder(args.semantic_model), batch_size=args.batch_size
        )
    rows, metrics = evaluate_answer_records(questions, generations, semantic_scorer=scorer)
    paths = write_answer_artifacts(args.run_dir, rows, metrics)
    summary = (
        f"answers={metrics['answerable_evaluated_count']} "
        f"em={metrics['exact_match']:.4f} f1={metrics['token_f1']:.4f}"
    )
    if "semantic_similarity_mean" in metrics:
        summary += f" semantic={metrics['semantic_similarity_mean']:.4f}"
    print(summary)
    print(f"Artifacts: {paths['answer_evaluation']}, {paths['answer_metrics']}")


if __name__ == "__main__":
    main()
