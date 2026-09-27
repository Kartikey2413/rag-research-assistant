"""Command-line interface for the RAG research assistant.

Example:
    python cli.py "What is retrieval-augmented generation?"
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from src.qa_engine import RAGPipeline

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Research Assistant")
    parser.add_argument("question", help="Question to ask your documents")
    parser.add_argument("--docs", default="data/sample_docs", help="Directory of documents to index")
    parser.add_argument(
        "--backend",
        default=os.getenv("EMBEDDING_BACKEND", "tfidf"),
        choices=["tfidf", "sentence-transformers"],
    )
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    pipeline = RAGPipeline(docs_dir=args.docs, backend=args.backend, top_k=args.top_k)
    n_chunks = pipeline.build()
    print(f"Indexed {n_chunks} chunks from {args.docs}\n")

    result = pipeline.ask(args.question)
    print("Answer:\n" + result["answer"] + "\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score {s['score']}): {s['snippet']}...")


if __name__ == "__main__":
    main()
