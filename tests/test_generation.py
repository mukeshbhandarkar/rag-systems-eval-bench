import json
from pathlib import Path

import torch
import pytest

from ragbench.experiments.runner import run_experiment
from ragbench.generation.base import Generator
from ragbench.generation.hf import HuggingFaceGenerator
from ragbench.generation.prompting import build_budgeted_context, format_prompt
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.hybrid import HybridRetriever
from ragbench.schemas import BenchmarkQuestion, Chunk, GeneratedAnswer, RetrievalResult


class WordTokenizer:
    def __init__(self) -> None:
        self.words: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.words:
                token_id = len(self.words) + 1
                self.words[word] = token_id
                self.reverse[token_id] = word
            ids.append(self.words[word])
        return [0, *ids] if add_special_tokens else ids

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if values == [101, 102]:
            return "known answer"
        return " ".join(self.reverse[value] for value in values if value != 0)

    def __call__(self, text: str, *, return_tensors: str, truncation: bool):
        return {"input_ids": torch.tensor([self.encode(text)])}


class FakeSeq2SeqModel:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.device = None
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        assert kwargs["do_sample"] is False
        return torch.tensor([[101, 102]])


class FakeGenerator(Generator):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake/generator"

    def generate(self, question: str, ranked_chunks: list[Chunk]) -> GeneratedAnswer:
        self.calls += 1
        context = "\n".join(f"[{rank}] {chunk.text}" for rank, chunk in enumerate(ranked_chunks, 1))
        return GeneratedAnswer(
            answer="fixed answer",
            prompt=format_prompt(question, context),
            context=context,
            context_chunk_ids=[chunk.chunk_id for chunk in ranked_chunks],
            input_token_count=10,
            output_token_count=2,
            truncated=False,
        )


class StaticRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def fit(self, chunks: list[Chunk]) -> None:
        pass

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [RetrievalResult(chunk_id=value, score=1.0, rank=rank) for rank, value in enumerate(self.ids[:top_k], 1)]


def test_prompt_format_is_exact() -> None:
    assert format_prompt("What is alpha?", "[1] Alpha is first.") == (
        "Answer the question using only the provided context.\n"
        "If the answer is not supported by the context, say you do not know.\n\n"
        "Context:\n[1] Alpha is first.\n\n"
        "Question:\nWhat is alpha?\n\nAnswer:"
    )


def test_context_preserves_rank_and_truncates_to_token_budget() -> None:
    tokenizer = WordTokenizer()
    chunks = [Chunk("first", "d", "one two", 0), Chunk("second", "d", "three four", 1)]
    desired = format_prompt("question", "[1] one two\n[2] three")
    budget = len(tokenizer.encode(desired, add_special_tokens=True))
    context, ids, count, truncated = build_budgeted_context(
        question="question", ranked_chunks=chunks, tokenizer=tokenizer, max_input_tokens=budget
    )
    assert context == "[1] one two\n[2] three"
    assert ids == ["first", "second"]
    assert count <= budget
    assert truncated is True


def test_hf_generator_is_cpu_greedy_and_reused_for_questions() -> None:
    tokenizer = WordTokenizer()
    model = FakeSeq2SeqModel()
    generator = HuggingFaceGenerator(
        model_name="fake/flan", tokenizer=tokenizer, model=model,
        max_input_tokens=100, max_new_tokens=7
    )
    chunk = Chunk("c", "d", "supporting context", 0)
    first = generator.generate("question one", [chunk])
    second = generator.generate("question two", [chunk])
    assert first.answer == second.answer == "known answer"
    assert model.device == "cpu" and model.eval_called
    assert model.generate_calls == 2
    assert first.output_token_count == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model_name": ""}, "model_name"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_input_tokens": 0}, "max_input_tokens"),
    ],
)
def test_generator_rejects_invalid_configuration(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HuggingFaceGenerator(tokenizer=WordTokenizer(), model=FakeSeq2SeqModel(), **kwargs)


def test_generation_artifact_with_bm25_and_retrieval_only_regression(tmp_path: Path) -> None:
    chunks = [Chunk("c1", "d", "alpha unique evidence", 0), Chunk("c2", "d", "beta", 1)]
    questions = [BenchmarkQuestion("q1", "alpha unique", ["c1"])]
    generator = FakeGenerator()
    generated = run_experiment(
        chunks=chunks, questions=questions, retriever=BM25Retriever(), top_k=1,
        run_id="generated", corpus_version="c", benchmark_version="b",
        output_root=tmp_path, generator=generator,
        generation_config={"model_name": generator.model_name, "do_sample": False},
    )
    row = json.loads((tmp_path / "generated" / "generation.jsonl").read_text())
    assert generator.calls == 1
    assert row["generated_answer"] == "fixed answer"
    assert row["retrieved_chunk_ids"] == row["context_chunk_ids"] == ["c1"]
    assert set(generated.output_paths) == {"config", "retrieval", "metrics", "generation"}

    retrieval_only = run_experiment(
        chunks=chunks, questions=questions, retriever=BM25Retriever(), top_k=1,
        run_id="retrieval-only", corpus_version="c", benchmark_version="b", output_root=tmp_path,
    )
    assert set(retrieval_only.output_paths) == {"config", "retrieval", "metrics"}
    assert not (tmp_path / "retrieval-only" / "generation.jsonl").exists()


def test_generator_is_independent_of_hybrid_retrieval(tmp_path: Path) -> None:
    chunks = [Chunk("c1", "d", "one", 0), Chunk("c2", "d", "two", 1)]
    hybrid = HybridRetriever(
        lexical_retriever=StaticRetriever(["c1", "c2"]),
        semantic_retriever=StaticRetriever(["c2", "c1"]),
        candidate_k=2,
    )
    generator = FakeGenerator()
    result = run_experiment(
        chunks=chunks,
        questions=[BenchmarkQuestion("q", "question", ["c1"])],
        retriever=hybrid,
        retriever_name="hybrid",
        generator=generator,
        top_k=2,
        run_id="hybrid-generation",
        corpus_version="c",
        benchmark_version="b",
        output_root=tmp_path,
    )
    assert generator.calls == 1
    assert "generation" in result.output_paths
