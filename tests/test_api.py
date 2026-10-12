"""API tests. Fully offline: tfidf backend, no Ollama needed."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["RATE_LIMIT"] = os.environ["UPLOAD_RATE_LIMIT"] = "1000/minute"  # before api.main is imported

CATS = b"Cats are small carnivorous mammals. Cats purr when content and sleep most of the day. " * 10
DOGS = b"Dogs are loyal domesticated animals. Dogs bark to communicate and love long walks. " * 10


@pytest.fixture()
def client(tmp_path, monkeypatch):
    (tmp_path / "rag.txt").write_text("Retrieval-augmented generation pairs a retriever with a language model. " * 10)
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # unreachable -> extractive fallback
    monkeypatch.setenv("MAX_SESSION_FILES", "3")
    from api.main import app

    with TestClient(app) as c:
        yield c


def upload(client, *files):
    return client.post("/upload", files=[("files", (n, d, "text/plain")) for n, d in files])


def ask(client, question, session_id=None):
    return client.post("/ask", json={"question": question, "session_id": session_id})


def test_health_and_index_page(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0 and body["mode"] == "extractive"
    assert "RAG Research Assistant" in client.get("/").text


def test_ask_shared_corpus(client):
    body = ask(client, "What is retrieval-augmented generation?").json()
    assert body["scope"] == "shared" and "rag.txt" in body["sources"][0]["source"]


def test_ask_validation(client):
    assert ask(client, "hi").status_code == 422


def test_upload_then_ask_uses_only_uploaded_files(client):
    r = upload(client, ("cats.txt", CATS))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"] == ["cats.txt"] and body["chunks"] > 0

    ans = ask(client, "Why do cats purr?", body["session_id"]).json()
    assert ans["scope"] == "session"
    assert ans["sources"] and all("cats.txt" in s["source"] for s in ans["sources"])


def test_sessions_are_isolated(client):
    a = upload(client, ("cats.txt", CATS)).json()["session_id"]
    b = upload(client, ("dogs.txt", DOGS)).json()["session_id"]
    ans = ask(client, "Tell me about cats", b).json()
    assert a != b and all("dogs.txt" in s["source"] for s in ans["sources"])


def test_upload_validation(client):
    assert upload(client, ("evil.exe", b"MZ")).status_code == 415
    assert upload(client, ("empty.txt", b"  ")).status_code == 422
    assert upload(client, *[(f"f{i}.txt", CATS) for i in range(4)]).status_code == 413


def test_unknown_or_deleted_session(client):
    assert ask(client, "anything?", "nope").status_code == 404
    sid = upload(client, ("cats.txt", CATS)).json()["session_id"]
    assert client.delete(f"/session/{sid}").status_code == 200
    assert ask(client, "Why do cats purr?", sid).status_code == 404
