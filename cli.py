"""Command line: python cli.py "What is retrieval-augmented generation?" """
import argparse
import os

from dotenv import load_dotenv

from src.qa_engine import RAGPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG Research Assistant")
    parser.add_argument("question")
    parser.add_argument("--docs", default="data/sample_docs", help="folder of .txt/.md/.pdf files")
    parser.add_argument("--backend", default=os.getenv("EMBEDDING_BACKEND", "tfidf"), choices=["tfidf", "sentence-transformers"])
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    pipeline = RAGPipeline(docs_dir=args.docs, backend=args.backend, top_k=args.top_k)
    print(f"Indexed {pipeline.build()} chunks from {args.docs}\n")

    result = pipeline.ask(args.question)
    print(f"Answer:\n{result['answer']}\n\nSources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (score {s['score']}): {s['snippet']}...")


if __name__ == "__main__":
    main()
