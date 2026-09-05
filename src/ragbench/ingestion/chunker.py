"""Deterministic dependency-free word chunking."""

from __future__ import annotations

from ragbench.schemas import Chunk, Document


def chunk_documents(
    documents: list[Document], chunk_size_words: int, overlap_words: int = 0
) -> list[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        for position, start in enumerate(range(0, len(words), step)):
            window = words[start : start + chunk_size_words]
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::chunk-{position:04d}",
                    doc_id=document.doc_id,
                    text=" ".join(window),
                    position=position,
                    metadata={"word_start": start, "word_count": len(window)},
                )
            )
            if start + chunk_size_words >= len(words):
                break
    return chunks
