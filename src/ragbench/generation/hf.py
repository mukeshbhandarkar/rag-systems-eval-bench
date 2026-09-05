"""Deterministic CPU generation with Hugging Face sequence-to-sequence models."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ragbench.generation.base import Generator
from ragbench.generation.prompting import build_budgeted_context, format_prompt
from ragbench.schemas import Chunk, GeneratedAnswer


DEFAULT_GENERATOR_MODEL = "google/flan-t5-small"


class HuggingFaceGenerator(Generator):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GENERATOR_MODEL,
        max_new_tokens: int = 64,
        max_input_tokens: int = 512,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must either both be supplied or both be omitted")
        self._model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        try:
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            self.model = model or AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            raise ValueError(f"Could not load generator model {model_name!r}: {exc}") from exc
        self.model.to("cpu")
        self.model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, question: str, ranked_chunks: list[Chunk]) -> GeneratedAnswer:
        context, context_ids, input_count, truncated = build_budgeted_context(
            question=question,
            ranked_chunks=ranked_chunks,
            tokenizer=self.tokenizer,
            max_input_tokens=self.max_input_tokens,
        )
        prompt = format_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        inputs = {name: tensor.to("cpu") for name, tensor in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return GeneratedAnswer(
            answer=answer,
            prompt=prompt,
            context=context,
            context_chunk_ids=context_ids,
            input_token_count=input_count,
            output_token_count=int(output[0].numel()),
            truncated=truncated,
        )
