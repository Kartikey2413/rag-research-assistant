# RAG Research Assistant

![CI](https://github.com/Kartikey2413/rag-research-assistant/actions/workflows/ci.yml/badge.svg)

Upload your documents, let the assistant study them, then ask questions. Every answer comes with the exact passages it was drawn from. Built with FastAPI, a local LLM via [Ollama](https://ollama.com), Docker, and an evaluation harness that measures retrieval quality.

**Live demo:** `http://<your-ec2-public-ip>` · **API docs:** `/docs`

## Results

Measured on a hand-written question set (`evaluation/eval_set.json`):

<!-- EVAL:START -->
_Run the evaluation to fill in this table._
<!-- EVAL:END -->

Hit@k: the correct source is in the top-k passages. MRR: how high it ranks (1.0 = always first). Keyword score: answers containing at least half of the expected key terms.

## How it works

```
upload ─▶ chunk ─▶ index (tf-idf | sentence-transformers) ─┐
question ─▶ retrieve top-k passages ─▶ Ollama LLM ─▶ answer + sources
```

1. **Ingest** `.txt`, `.md`, `.pdf` into overlapping chunks (`src/ingest.py`).
2. **Index** with `tfidf` (default, offline) or `sentence-transformers` (better on paraphrased queries) (`src/embed_store.py`).
3. **Generate** from the retrieved passages with Ollama, citing sources (`src/llm_client.py`). If Ollama isn't running or the model isn't pulled, it returns the best passages directly instead of failing.

Each visitor's uploads live in a private session (limited files and size, deleted after 60 minutes idle), so users never see each other's documents.

## Quickstart

```bash
pip install -r requirements-dev.txt
ollama pull llama3                 # optional: LLM answers; otherwise extractive fallback
uvicorn api.main:app               # web UI + API on http://localhost:8000
```

Other entry points: `streamlit run app.py` · `python cli.py "your question" --docs data/sample_docs`

**Docker**

```bash
cp .env.example .env
docker compose up --build          # http://localhost:8000
```

The container reaches Ollama on your machine at `host.docker.internal:11434`. On Linux, start Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve` so the container can connect.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Multipart `files` → indexes them in a private session, returns `session_id` |
| `POST` | `/ask` | `{"question", "session_id"?}` → answer, sources, latency. No `session_id` searches the sample corpus |
| `DELETE` | `/session/{id}` | Delete a session and its files |
| `GET` | `/health` | Index status, chunk count, `llm` or `extractive` mode |

```bash
curl -F "files=@notes.pdf" localhost:8000/upload
curl -H 'Content-Type: application/json' localhost:8000/ask \
  -d '{"question": "What is the main finding?", "session_id": "<id from upload>"}'
```

Optional env vars: `OLLAMA_MODEL`, `OLLAMA_HOST`, `EMBEDDING_BACKEND`, `TOP_K`, `RATE_LIMIT` (20/minute), `UPLOAD_RATE_LIMIT` (5/minute), `SESSION_TTL_MIN` (60), `MAX_SESSIONS` (20), `MAX_SESSION_FILES` (5), `MAX_UPLOAD_MB` (5).

## Evaluation

Replace the template in `evaluation/eval_set.json` with 25–40 questions over your documents (`expected_source` = part of the file name, `expected_keywords` = terms a correct answer contains; include paraphrased questions), then:

```bash
python -m evaluation.run_eval --backends tfidf sentence-transformers --update-readme
```

Add `--with-llm` to grade real Ollama answers. Without it, retrieval is measured alone (free and deterministic).

## Deploy

EC2 + Docker guide: [`DEPLOY.md`](DEPLOY.md). A free-tier instance can't run an Ollama model, so a public demo there runs in extractive mode.

## Structure

```
api/            FastAPI app and web UI
src/            ingest, embed_store, llm_client, qa_engine, llm_status
evaluation/     eval set and metrics runner
tests/          offline tests (pytest tests/ -v)
deploy/         EC2 bootstrap script
app.py cli.py   Streamlit UI and command line
```

## License

MIT, see `LICENSE`.
