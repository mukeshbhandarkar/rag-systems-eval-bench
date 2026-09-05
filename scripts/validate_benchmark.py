#!/usr/bin/env python3
"""Validate the local public benchmark corpus, annotations, and versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragbench.benchmark.validation import validate_public_benchmark
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import BenchmarkQuestion, Chunk


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {path}: {exc}") from exc


def _load_jsonl(path: Path, record_type: type):
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(record_type(**json.loads(line)))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid {path} line {line_number}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/public/raw"))
    parser.add_argument("--chunks", type=Path, default=Path("data/public/processed/chunks.jsonl"))
    parser.add_argument(
        "--corpus-metadata",
        type=Path,
        default=Path("data/public/processed/corpus_metadata.json"),
    )
    parser.add_argument(
        "--questions", type=Path, default=Path("data/public/benchmark/questions.jsonl")
    )
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        default=Path("data/public/benchmark/benchmark_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_public_benchmark(
        documents=load_documents(args.raw_dir),
        chunks=_load_jsonl(args.chunks, Chunk),
        questions=_load_jsonl(args.questions, BenchmarkQuestion),
        corpus_metadata=_load_json(args.corpus_metadata),
        benchmark_metadata=_load_json(args.benchmark_metadata),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
