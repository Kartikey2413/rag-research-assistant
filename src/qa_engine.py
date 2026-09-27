"""Ties ingestion, retrieval, and generation into a single pipeline."""
from __future__ import annotations

from pathlib import Path

from src.embed_store import VectorStore
from src.ingest import build_corpus
from src.llm_client import generate_answer


class RAGPipeline:
    """End-to-end retrieval-augmented Q&A over a folder of documents."""

    def __init__(self, docs_dir: str | Path, backend: str = "tfidf", top_k: int = 4):
        self.docs_dir = Path(docs_dir)
        self.top_k = top_k
        self.store = VectorStore(backend=backend)
        self._built = False

    def build(self) -> int:
        """Ingest and index the document directory. Returns the chunk count."""
        chunks = build_corpus(self.docs_dir)
        self.store.build(chunks)
        self._built = True
        return len(chunks)

    def ask(self, query: str) -> dict:
        if not self._built:
            self.build()
        hits = self.store.search(query, k=self.top_k)
        answer = generate_answer(query, hits)
        return {
            "answer": answer,
            "sources": [
                {"source": c.source, "snippet": c.text[:200], "score": round(score, 3)}
                for c, score in hits
            ],
        }
