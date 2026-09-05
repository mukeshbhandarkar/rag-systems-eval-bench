from pathlib import Path

import pytest

from ragbench.ingestion.chunker import chunk_documents
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import Document


def test_loader_order_and_ids_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("last", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("ignored", encoding="utf-8")
    first = load_documents(tmp_path)
    second = load_documents(tmp_path)
    assert [doc.source for doc in first] == ["a.md", "z.txt"]
    assert [doc.doc_id for doc in first] == [doc.doc_id for doc in second]
    assert len(set(doc.doc_id for doc in first)) == 2


def test_chunking_is_deterministic_with_stable_ids_and_overlap() -> None:
    document = Document("doc", "Title", "one two three four five six", "doc.txt")
    chunks = chunk_documents([document], chunk_size_words=4, overlap_words=2)
    assert chunks == chunk_documents([document], chunk_size_words=4, overlap_words=2)
    assert [chunk.chunk_id for chunk in chunks] == ["doc::chunk-0000", "doc::chunk-0001"]
    assert [chunk.text for chunk in chunks] == ["one two three four", "three four five six"]
    assert [chunk.position for chunk in chunks] == [0, 1]


@pytest.mark.parametrize("size,overlap", [(0, 0), (3, -1), (3, 3), (3, 4)])
def test_invalid_chunk_configuration(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], size, overlap)
