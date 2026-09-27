# RAG Research Assistant

A retrieval-augmented Q&A system that answers questions grounded in your own
documents — with sources — instead of relying on a language model's memory
alone.

Point it at a folder of notes, papers, or policy documents, ask a question,
and it returns an answer along with the exact passages it drew from.

## Why this exists

Language models are fluent, but fluent isn't the same as correct — a model
can state something confidently that it never actually saw. Retrieval-
augmented generation (RAG) fixes this by making the model answer only from
passages a retriever actually found, and by surfacing those passages so a
reader can check the claim themselves.

## How it works

```
question ─▶ retriever (search your docs) ─▶ top-k passages ─▶ LLM ─▶ answer + citations
```

1. **Ingest** — documents (`.txt`, `.md`, `.pdf`) are split into overlapping
   chunks (`src/ingest.py`).
2. **Index** — chunks are turned into searchable vectors. Two backends are
   supported (`src/embed_store.py`):
   - `tfidf` (default) — classic sparse retrieval via scikit-learn. Zero
     external dependencies, runs anywhere, no internet required.
   - `sentence-transformers` — dense embeddings for better semantic recall
     on paraphrased queries. Downloads a small model on first use.
3. **Generate** — the top-matching chunks are passed to Claude, which
   answers using only that context and cites its sources (`src/llm_client.py`).
   If no API key is set, it falls back to returning the best-matching
   passages directly, so the project is still fully runnable with zero
   setup.

## Quickstart

```bash
git clone https://github.com/<your-username>/rag-research-assistant.git
cd rag-research-assistant
pip install -r requirements.txt

# optional — enables grounded LLM answers instead of the extractive fallback
cp .env.example .env  # then add your ANTHROPIC_API_KEY

# ask a question against the bundled sample docs
python cli.py "What is retrieval-augmented generation?"

# or launch the chat UI
streamlit run app.py
```

Drop your own `.txt` / `.md` / `.pdf` files into `data/sample_docs/` (or
point `--docs` / the sidebar at another folder) to ask questions about your
own material instead.

## Project structure

```
rag-research-assistant/
├── app.py                  # Streamlit chat UI
├── cli.py                  # command-line entry point
├── src/
│   ├── ingest.py           # document loading + chunking
│   ├── embed_store.py      # tfidf / sentence-transformers retrieval
│   ├── llm_client.py       # Claude-backed generation + extractive fallback
│   └── qa_engine.py        # wires the pipeline together
├── data/sample_docs/       # example corpus so the demo works out of the box
├── tests/                  # offline smoke tests (no API key needed)
├── requirements.txt
└── .env.example
```

## Running the tests

```bash
pytest tests/ -v
```

Tests run entirely on the `tfidf` backend with no API key or internet
access required, so they're safe to run in CI.

## Design notes

- **Two retrieval backends, one interface** — `tfidf` is the safe default
  so the project works offline out of the box; swapping in
  `sentence-transformers` is a one-line config change when better recall is
  worth the extra dependency.
- **Graceful degradation over hard failure** — with no `ANTHROPIC_API_KEY`
  set, the assistant still answers (extractively) rather than crashing,
  which matters for anyone cloning this without a key on hand.
- **Chunking is intentionally simple** — a fixed-size sliding window keeps
  the pipeline predictable and dependency-free; a token-aware or
  semantic-boundary splitter would be the natural next upgrade for a larger
  or more heterogeneous corpus.

## Possible extensions

- Swap the pickle-based `VectorStore` for FAISS or a hosted vector DB to
  scale past a few thousand chunks.
- Add re-ranking of retrieved chunks before generation.
- Support multi-turn follow-up questions that reference earlier answers.
- Track which cited sources users actually click, as a feedback signal for
  retrieval quality.

## License

MIT — see `LICENSE`.
