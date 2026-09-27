"""LLM-backed answer generation, with a dependency-free extractive fallback."""
from __future__ import annotations

import os

from src.ingest import Chunk

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the "
    "provided context. Cite sources inline as [source]. If the context "
    "doesn't contain the answer, say so plainly instead of guessing."
)


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)


def _extractive_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Fallback used when Ollama isn't reachable or no hits are found.

    Returns the highest-scoring passages directly so the pipeline stays
    fully usable even if the local model server is down.
    """
    if not hits:
        return "No relevant passages were found for that question."
    lines = [f'Based on the most relevant passages found for: "{query}"\n']
    for chunk, score in hits:
        lines.append(f"- ({chunk.source}, relevance {score:.2f}): {chunk.text[:280]}...")
    return "\n".join(lines)


def generate_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Generate a grounded answer using a local Ollama model, falling back
    to an extractive answer built from the top chunks if Ollama isn't
    running, isn't reachable, or the request otherwise fails.
    """
    if not hits:
        return _extractive_answer(query, hits)

    import requests  # imported lazily so the extractive path has no hard dependency

    context = _format_context(hits)
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "stream": False,
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.exceptions.ConnectionError:
        # Ollama isn't running — degrade gracefully instead of crashing.
        fallback = _extractive_answer(query, hits)
        return (
            f"[Ollama not reachable at {OLLAMA_HOST} — is it running? "
            "Falling back to extractive answer.]\n\n" + fallback
        )
    except Exception as e:
        fallback = _extractive_answer(query, hits)
        return f"[Ollama request failed: {e} — falling back to extractive answer.]\n\n" + fallback