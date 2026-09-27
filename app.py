"""Streamlit UI for the RAG Research Assistant.

Run with:
    streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

from src.qa_engine import RAGPipeline

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", page_icon="🔎")
st.title("🔎 RAG Research Assistant")
st.caption("Ask questions and get answers grounded in your own documents, with sources.")

with st.sidebar:
    st.header("Settings")
    docs_dir = st.text_input("Documents folder", value="data/sample_docs")
    backend = st.selectbox("Embedding backend", ["tfidf", "sentence-transformers"], index=0)
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    st.write("LLM mode:", "Claude (grounded)" if has_key else "Extractive fallback (no API key set)")


@st.cache_resource(show_spinner="Indexing documents...")
def get_pipeline(docs_dir: str, backend: str) -> RAGPipeline:
    pipeline = RAGPipeline(docs_dir=docs_dir, backend=backend)
    pipeline.build()
    return pipeline


pipeline = get_pipeline(docs_dir, backend)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = pipeline.ask(question)
    with st.chat_message("assistant"):
        st.write(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"**{s['source']}** (score {s['score']})  \n{s['snippet']}...")

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
