#!/usr/bin/env python3
"""Audit local release packages for consistency and accidental private data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ragbench.benchmark.validation import validate_public_benchmark
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import BenchmarkQuestion, Chunk


ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://huggingface.co/datasets/Max00035/rag-systems-eval-benchmark"
SPACE_URL = "https://huggingface.co/spaces/Max00035/rag-systems-eval-bench"
GITHUB_URL = "https://github.com/mukeshbhandarkar/rag-systems-eval-bench"
PUBLISHABLE_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "hf_dataset",
    ROOT / "hf_space",
    ROOT / "benchmark_results",
    ROOT / "data/public",
)
FORBIDDEN_TEXT = {
    "fixture-v0.1": "stale fixture version",
    "license: other": "stale dataset license metadata",
    "Dataset URL — add after publication": "dataset URL placeholder",
    "Space URL — add after publication": "Space URL placeholder",
    "GitHub URL — add after publication": "GitHub URL placeholder",
    "http://localhost": "localhost URL",
    "https://localhost": "localhost URL",
    "http://127.0.0.1": "loopback URL",
    "https://127.0.0.1": "loopback URL",
    "/home/mack": "private absolute path",
    "/.cache/huggingface": "model cache path",
}
SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def audit_release(root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    publishable_files = list(_files(PUBLISHABLE_ROOTS))
    for path in publishable_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle, label in FORBIDDEN_TEXT.items():
            if needle in text:
                errors.append(f"{path.relative_to(root)} contains {label}: {needle!r}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"{path.relative_to(root)} contains credential-like text matching {pattern.pattern!r}"
                )

    dataset_readme = (root / "hf_dataset/README.md").read_text(encoding="utf-8")
    space_readme = (root / "hf_space/README.md").read_text(encoding="utf-8")
    dashboard = (root / "hf_space/index.html").read_text(encoding="utf-8")
    if "license: cc-by-4.0" not in dataset_readme:
        errors.append("Dataset card does not declare license: cc-by-4.0")
    if "license: mit" not in space_readme:
        errors.append("Space card does not declare license: mit")
    for label, url, texts in (
        ("Dataset", DATASET_URL, (dataset_readme, space_readme, dashboard)),
        ("Space", SPACE_URL, (dataset_readme, dashboard)),
        ("GitHub", GITHUB_URL, (dataset_readme, space_readme, dashboard)),
    ):
        if any(url not in text for text in texts):
            errors.append(f"Real {label} URL is missing from a required publishable file")

    json_count = 0
    jsonl_count = 0
    for directory in (root / "hf_dataset", root / "hf_space/data"):
        for path in sorted(directory.glob("*.json")):
            _parse_json(path, errors)
            json_count += 1
        for path in sorted(directory.glob("*.jsonl")):
            _parse_jsonl(path, errors)
            jsonl_count += 1

    documents = load_documents(root / "data/public/raw")
    chunks = _records(root / "data/public/processed/chunks.jsonl", Chunk)
    questions = _records(root / "data/public/benchmark/questions.jsonl", BenchmarkQuestion)
    benchmark_summary = validate_public_benchmark(
        documents=documents,
        chunks=chunks,
        questions=questions,
        corpus_metadata=json.loads(
            (root / "data/public/processed/corpus_metadata.json").read_text(encoding="utf-8")
        ),
        benchmark_metadata=json.loads(
            (root / "data/public/benchmark/benchmark_metadata.json").read_text(encoding="utf-8")
        ),
    )

    exported_questions = _jsonl_values(root / "hf_dataset/benchmark.jsonl")
    if len(exported_questions) != len(questions):
        errors.append("Exported dataset question count does not match public annotations")
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    for record in exported_questions:
        if not set(record.get("relevant_chunk_ids", [])) <= chunk_ids:
            errors.append(f"Exported question {record.get('question_id')} has invalid relevant chunks")

    dashboard_results = json.loads(
        (root / "hf_space/data/results.json").read_text(encoding="utf-8")
    )
    dashboard_benchmark = dashboard_results.get("benchmark", {})
    if dashboard_benchmark.get("benchmark_version") != "ragbench-v1":
        errors.append("Dashboard benchmark version is not ragbench-v1")
    if dashboard_benchmark.get("corpus_version") != "ragbench-v1":
        errors.append("Dashboard corpus version is not ragbench-v1")
    if dashboard_benchmark.get("categories") != benchmark_summary["categories"]:
        errors.append("Dashboard category counts do not match benchmark validation")

    if errors:
        raise ValueError("Release audit failed:\n- " + "\n- ".join(errors))
    return {
        "status": "ok",
        "publishable_file_count": len(publishable_files),
        "json_files_validated": json_count,
        "jsonl_files_validated": jsonl_count,
        "benchmark": benchmark_summary,
        "licenses": {"source_and_space": "MIT", "dataset": "CC BY 4.0"},
        "remaining_placeholders": [],
    }


def _files(paths):
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (item for item in path.rglob("*") if item.is_file())


def _parse_json(path: Path, errors: list[str]) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid JSON in {path.relative_to(ROOT)}: {exc}")


def _parse_jsonl(path: Path, errors: list[str]) -> None:
    try:
        _jsonl_values(path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid JSONL in {path.relative_to(ROOT)}: {exc}")


def _jsonl_values(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"line {line_number}: {exc.msg}", exc.doc, exc.pos
                ) from exc
    return values


def _records(path: Path, record_type):
    return [record_type(**value) for value in _jsonl_values(path)]


def main() -> None:
    print(json.dumps(audit_release(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
