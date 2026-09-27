"""LLM-backed answer generation, with a dependency-free extractive fallback."""
from __future__ import annotations

import os

from src.ingest import Chunk

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the "
    "provided context. Cite sources inline as [source]. If the context "
    "doesn't contain the answer, say so plainly instead of guessing."
)


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)


def _extractive_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Fallback used when no LLM API key is configured.

    Returns the highest-scoring passages directly so the pipeline stays
    fully usable offline, rather than failing when a key is missing.
    """
    if not hits:
        return "No relevant passages were found for that question."
    lines = [f'Based on the most relevant passages found for: "{query}"\n']
    for chunk, score in hits:
        lines.append(f"- ({chunk.source}, relevance {score:.2f}): {chunk.text[:280]}...")
    return "\n".join(lines)


def generate_answer(query: str, hits: list[tuple[Chunk, float]]) -> str:
    """Generate a grounded answer using Anthropic's API if a key is set,
    otherwise fall back to an extractive answer built from the top chunks.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _extractive_answer(query, hits)

    import anthropic  # imported lazily so the extractive path has no hard dependency

    client = anthropic.Anthropic(api_key=api_key)
    context = _format_context(hits)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(block.text for block in message.content if block.type == "text")
