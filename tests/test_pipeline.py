"""Smoke tests that run fully offline (tfidf backend, no API key needed)."""
from src.qa_engine import RAGPipeline


def test_pipeline_builds_and_answers(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text(
        "Retrieval-augmented generation combines a retriever with a language "
        "model so answers stay grounded in real documents instead of "
        "memorized text."
    )
    (docs_dir / "b.txt").write_text(
        "The data science lifecycle typically covers problem framing, data "
        "collection, cleaning, modeling, evaluation, and deployment."
    )

    pipeline = RAGPipeline(docs_dir=docs_dir, backend="tfidf")
    n_chunks = pipeline.build()
    assert n_chunks > 0

    result = pipeline.ask("What is retrieval-augmented generation?")
    assert result["answer"]
    assert result["sources"]
    assert result["sources"][0]["source"] == "a.txt"


def test_empty_directory_returns_no_sources(tmp_path):
    docs_dir = tmp_path / "empty"
    docs_dir.mkdir()

    pipeline = RAGPipeline(docs_dir=docs_dir, backend="tfidf")
    pipeline.build()

    result = pipeline.ask("Anything in here?")
    assert result["sources"] == []
