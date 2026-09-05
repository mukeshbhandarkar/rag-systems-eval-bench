"""Inspectable prompt and token-budget-aware context construction."""

from __future__ import annotations

from typing import Protocol, Sequence

from ragbench.schemas import Chunk


PROMPT_TEMPLATE = """Answer the question using only the provided context.
If the answer is not supported by the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""


class TokenCodec(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...


def format_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip(), context=context)


def _context_entry(rank: int, text: str) -> str:
    return f"[{rank}] {text.strip()}"


def build_budgeted_context(
    *,
    question: str,
    ranked_chunks: Sequence[Chunk],
    tokenizer: TokenCodec,
    max_input_tokens: int,
) -> tuple[str, list[str], int, bool]:
    """Pack ranked chunks into a prompt-sized budget, truncating only the last one."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be greater than zero")
    empty_count = len(tokenizer.encode(format_prompt(question, ""), add_special_tokens=True))
    if empty_count > max_input_tokens:
        raise ValueError("max_input_tokens is too small for the instruction and question")

    entries: list[str] = []
    chunk_ids: list[str] = []
    truncated = False
    for rank, chunk in enumerate(ranked_chunks, start=1):
        full_entry = _context_entry(rank, chunk.text)
        candidate_entries = [*entries, full_entry]
        candidate_context = "\n".join(candidate_entries)
        candidate_count = len(
            tokenizer.encode(format_prompt(question, candidate_context), add_special_tokens=True)
        )
        if candidate_count <= max_input_tokens:
            entries = candidate_entries
            chunk_ids.append(chunk.chunk_id)
            continue

        token_ids = tokenizer.encode(chunk.text.strip(), add_special_tokens=False)
        low, high = 0, len(token_ids)
        best_entry: str | None = None
        while low <= high:
            midpoint = (low + high) // 2
            prefix = tokenizer.decode(token_ids[:midpoint], skip_special_tokens=True).strip()
            entry = _context_entry(rank, prefix) if prefix else ""
            trial_context = "\n".join([*entries, entry] if entry else entries)
            trial_count = len(
                tokenizer.encode(format_prompt(question, trial_context), add_special_tokens=True)
            )
            if entry and trial_count <= max_input_tokens:
                best_entry = entry
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_entry is not None:
            entries.append(best_entry)
            chunk_ids.append(chunk.chunk_id)
        truncated = True
        break

    if len(chunk_ids) < len(ranked_chunks):
        truncated = True
    context = "\n".join(entries)
    input_token_count = len(
        tokenizer.encode(format_prompt(question, context), add_special_tokens=True)
    )
    return context, chunk_ids, input_token_count, truncated
