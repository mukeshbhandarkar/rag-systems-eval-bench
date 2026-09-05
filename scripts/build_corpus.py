#!/usr/bin/env python3
"""Build deterministic processed chunks from local raw documents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ragbench.ingestion.chunker import chunk_documents
from ragbench.ingestion.loader import load_documents
from ragbench.schemas import to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--corpus-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.raw_dir)
    if not documents:
        raise SystemExit(f"No .txt or .md documents found in {args.raw_dir}")
    chunks = chunk_documents(documents, args.chunk_size, args.overlap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(to_dict(chunk), sort_keys=True) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    metadata = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_size_words": args.chunk_size,
        "overlap_words": args.overlap,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_version": args.corpus_version,
    }
    (args.output_dir / "corpus_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Built {len(chunks)} chunks from {len(documents)} documents: {chunks_path}")


if __name__ == "__main__":
    main()
