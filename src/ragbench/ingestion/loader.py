"""Load local text documents in a deterministic order."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragbench.schemas import Document


SUPPORTED_SUFFIXES = {".txt", ".md"}


def _document_id(relative_path: Path) -> str:
    normalized = relative_path.as_posix()
    slug = re.sub(r"[^a-z0-9]+", "-", relative_path.stem.lower()).strip("-") or "document"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def load_documents(raw_dir: str | Path) -> list[Document]:
    """Load every .txt and .md file below *raw_dir* using stable IDs."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw document path is not a directory: {root}")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    documents: list[Document] = []
    for path in paths:
        relative_path = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Could not read UTF-8 document {path}: {exc}") from exc
        documents.append(
            Document(
                doc_id=_document_id(relative_path),
                title=path.stem,
                text=text,
                source=relative_path.as_posix(),
                metadata={"file_type": path.suffix.lower(), "relative_path": relative_path.as_posix()},
            )
        )
    return documents
