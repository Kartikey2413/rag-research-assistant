"""Vector store abstraction with two interchangeable retrieval backends."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingest import Chunk


class VectorStore:
    """Retrieves the most relevant chunks for a query.

    backend="tfidf" (default) runs fully offline with no external model
    download — good for small local corpora and for running this repo
    anywhere without setup friction.

    backend="sentence-transformers" gives better semantic recall on larger
    or more paraphrased corpora, at the cost of downloading a model the
    first time it runs.
    """

    def __init__(self, backend: str = "tfidf"):
        self.backend = backend
        self.chunks: list[Chunk] = []
        self._vectorizer = None
        self._matrix = None
        self._model = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self._matrix = None
            return
        texts = [c.text for c in chunks]
        if self.backend == "tfidf":
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
            self._matrix = self._vectorizer.fit_transform(texts)
        elif self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer  # lazy import

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._matrix = self._model.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks or self._matrix is None:
            return []
        if self.backend == "tfidf":
            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix).flatten()
        else:
            q_vec = self._model.encode([query], normalize_embeddings=True)
            scores = (self._matrix @ q_vec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "VectorStore":
        with open(path, "rb") as f:
            return pickle.load(f)
