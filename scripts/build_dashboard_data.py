#!/usr/bin/env python3
"""Build dashboard and Hugging Face Dataset data from persisted benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragbench.dashboard.build import build_dashboard_payloads, write_public_packages
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import BenchmarkQuestion, Chunk


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path, record_type=None):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [record_type(**row) for row in rows] if record_type else rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-run", type=Path, required=True)
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--hybrid-run", type=Path, required=True)
    parser.add_argument("--generation-run", type=Path)
    parser.add_argument("--resources", type=Path, default=Path("configs/m6_resources.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/public/raw"))
    parser.add_argument("--chunks", type=Path, default=Path("data/public/processed/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/public/benchmark/questions.jsonl"))
    parser.add_argument("--benchmark-version", default="ragbench-v1")
    parser.add_argument("--corpus-version", default="ragbench-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_jsonl(args.questions, BenchmarkQuestion)
    chunks = load_jsonl(args.chunks, Chunk)
    runs = {
        "bm25": load_jsonl(args.bm25_run / "retrieval.jsonl"),
        "dense": load_jsonl(args.dense_run / "retrieval.jsonl"),
        "hybrid": load_jsonl(args.hybrid_run / "retrieval.jsonl"),
    }
    generation_metrics = None
    generation_config = None
    if args.generation_run:
        generation_metrics = load_json(args.generation_run / "answer_metrics.json")
        generation_config = load_json(args.generation_run / "config.json").get("generation", {})
    results, failures = build_dashboard_payloads(
        questions=questions,
        chunks=chunks,
        runs=runs,
        resources=load_json(args.resources),
        benchmark_version=args.benchmark_version,
        corpus_version=args.corpus_version,
        generation_metrics=generation_metrics,
        generation_config=generation_config,
    )
    write_public_packages(
        results=results,
        failures=failures,
        documents=load_documents(args.raw_dir),
        chunks=chunks,
        questions=questions,
        runs=runs,
        dashboard_data_dir=Path("hf_space/data"),
        dataset_dir=Path("hf_dataset"),
        benchmark_results_dir=Path("benchmark_results/ragbench-v1"),
    )
    print(
        f"Built dashboard data for {len(questions)} questions, {len(chunks)} chunks, "
        f"and {len(failures)} diagnostic failure cases"
    )


if __name__ == "__main__":
    main()
