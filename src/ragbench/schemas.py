"""Core data structures shared by ingestion, retrieval, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    gold_chunk_ids: list[str]
    reference_answers: list[str] = field(default_factory=list)
    answerable: bool = True
    metadata: Metadata = field(default_factory=dict)
    category: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    config: Metadata
    metrics: Metadata
    output_paths: dict[str, str]


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    prompt: str
    context: str
    context_chunk_ids: list[str]
    input_token_count: int
    output_token_count: int
    truncated: bool


def to_dict(value: object) -> dict[str, Any]:
    """Convert one of the project's dataclasses to a JSON-ready dictionary."""
    return asdict(value)
