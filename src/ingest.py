"""Document loading and chunking utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    """A single retrievable unit of text."""
    text: str
    source: str
    chunk_id: int


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(directory: str | Path) -> list[tuple[str, str]]:
    """Load every .txt, .md, and .pdf file in a directory.

    Returns a list of (source_name, full_text) tuples.
    """
    directory = Path(directory)
    docs: list[tuple[str, str]] = []
    for path in sorted(directory.glob("**/*")):
        if path.suffix.lower() in (".txt", ".md"):
            docs.append((path.name, _read_txt(path)))
        elif path.suffix.lower() == ".pdf":
            docs.append((path.name, _read_pdf(path)))
    return docs


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping character-based chunks.

    A simple sliding window keeps this dependency-free and predictable.
    Swap in a token-aware splitter (e.g. tiktoken) if you move to a larger
    or more heterogeneous corpus.
    """
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=chunk_id))
            chunk_id += 1
        start += chunk_size - overlap
    return chunks


def build_corpus(directory: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Load every document in a directory and return it as a flat list of chunks."""
    all_chunks: list[Chunk] = []
    for source, text in load_documents(directory):
        all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
    return all_chunks
