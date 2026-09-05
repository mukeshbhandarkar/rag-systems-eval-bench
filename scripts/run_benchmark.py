#!/usr/bin/env python3
"""Run a BM25 or dense retrieval benchmark and write versioned artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

from ragbench.experiments.runner import run_experiment
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.base import Retriever
from ragbench.schemas import BenchmarkQuestion, Chunk


T = TypeVar("T")


def load_jsonl(path: Path, record_type: type[T]) -> list[T]:
    records: list[T] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        records.append(record_type(**json.loads(line)))
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid record in {path} at line {line_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--benchmark-version", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generator-model", default="google/flan-t5-small")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    return parser.parse_args()


def build_retriever(args: argparse.Namespace) -> tuple[Retriever, dict[str, object]]:
    if args.retriever == "bm25":
        return BM25Retriever(k1=args.k1, b=args.b), {
            "k1": args.k1,
            "b": args.b,
            "tokenizer": "lowercase-word-regex",
        }

    # Importing lazily keeps the BM25 path independent of model initialization.
    from ragbench.retrieval.dense import DenseRetriever

    dense = DenseRetriever(model_name=args.model, batch_size=args.batch_size)
    dense_config: dict[str, object] = {
        "model_name": args.model,
        "embedding_dimension": dense.embedding_dimension,
        "batch_size": args.batch_size,
        "device": "cpu",
        "normalized_embeddings": True,
        "similarity": "dot_product_equivalent_to_cosine",
    }
    if args.retriever == "dense":
        return dense, dense_config

    from ragbench.retrieval.hybrid import HybridRetriever

    if args.candidate_k < args.top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k")
    hybrid = HybridRetriever(
        lexical_retriever=BM25Retriever(k1=args.k1, b=args.b),
        semantic_retriever=dense,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )
    return hybrid, {
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": args.rrf_k,
        "candidate_k": args.candidate_k,
        "lexical": {
            "retriever": "bm25",
            "k1": args.k1,
            "b": args.b,
            "tokenizer": "lowercase-word-regex",
        },
        "semantic": {"retriever": "dense", **dense_config},
    }


def main() -> None:
    args = parse_args()
    chunks = load_jsonl(args.chunks, Chunk)
    questions = load_jsonl(args.questions, BenchmarkQuestion)
    retriever, retriever_config = build_retriever(args)
    generator = None
    generation_config = None
    if args.generate:
        from ragbench.generation.hf import HuggingFaceGenerator

        generator = HuggingFaceGenerator(
            model_name=args.generator_model,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        generation_config = {
            "model_name": args.generator_model,
            "device": "cpu",
            "do_sample": False,
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "prompt_template": "context_only_v1",
        }
    result = run_experiment(
        chunks=chunks,
        questions=questions,
        retriever=retriever,
        top_k=args.top_k,
        run_id=args.run_id,
        corpus_version=args.corpus_version,
        benchmark_version=args.benchmark_version,
        output_root=args.runs_dir,
        retriever_name=args.retriever,
        retriever_config=retriever_config,
        generator=generator,
        generation_config=generation_config,
    )
    metrics = result.metrics
    print(
        f"{result.run_id}: queries={metrics['query_count']} top_k={metrics['top_k']} "
        f"recall={metrics['recall_at_k']:.4f} mrr={metrics['mrr_at_k']:.4f}"
    )
    print(f"Artifacts: {Path(result.output_paths['metrics']).parent}")


if __name__ == "__main__":
    main()
