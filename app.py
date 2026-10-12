"""Streamlit UI. Run with: streamlit run app.py"""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.llm_status import ollama_status
from src.qa_engine import RAGPipeline

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", page_icon="🔎")
st.title("🔎 RAG Research Assistant")
st.caption("Upload documents, then ask questions. Answers come with their sources.")

with st.sidebar:
    uploads = st.file_uploader("Upload documents", type=["txt", "md", "pdf"], accept_multiple_files=True)
    docs_dir = st.text_input("...or a documents folder", value="data/sample_docs", disabled=bool(uploads))
    backend = st.selectbox("Embedding backend", ["tfidf", "sentence-transformers"])
    st.write("LLM:", "Ollama" if ollama_status() == "llm" else "extractive fallback (Ollama unavailable)")

if uploads:  # save to a folder named by content hash so identical uploads are indexed once
    digest = hashlib.sha1(b"".join(f.name.encode() + f.getvalue() for f in uploads)).hexdigest()[:12]
    folder = Path(tempfile.gettempdir()) / f"rag_uploads_{digest}"
    if not folder.exists():
        folder.mkdir()
        for f in uploads:
            (folder / Path(f.name).name).write_bytes(f.getvalue())
    docs_dir = str(folder)


@st.cache_resource(show_spinner="Studying your documents...")
def get_pipeline(docs_dir: str, backend: str) -> RAGPipeline:
    pipeline = RAGPipeline(docs_dir=docs_dir, backend=backend)
    pipeline.build()
    return pipeline


pipeline = get_pipeline(docs_dir, backend)
history = st.session_state.setdefault("history", [])

for turn in history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if question := st.chat_input("Ask a question about your documents..."):
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = pipeline.ask(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"**{s['source']}** (score {s['score']})  \n{s['snippet']}...")
    history.append({"role": "assistant", "content": result["answer"]})
