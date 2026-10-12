"""Measure retrieval and answer quality on evaluation/eval_set.json.

    python -m evaluation.run_eval                      # tfidf, retrieval only (offline, deterministic)
    python -m evaluation.run_eval --backends tfidf sentence-transformers --update-readme
    python -m evaluation.run_eval --with-llm           # grade real Ollama answers

Hit@1 / Hit@k : expected source ranked first / within the top-k results
MRR           : mean reciprocal rank of the expected source
Keyword score : share of questions whose answer (or, without --with-llm, the retrieved
                passages) contains at least half of the expected keywords
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"


def evaluate(pipeline, items):
    ranks, keyword_ok, times = [], [], []
    for item in items:
        t = time.perf_counter()
        result = pipeline.ask(item["question"])
        times.append(time.perf_counter() - t)
        names = [str(s["source"]).lower() for s in result["sources"]]
        ranks.append(next((i for i, n in enumerate(names) if item["expected_source"].lower() in n), None))
        words = [k.lower() for k in item["expected_keywords"]]
        keyword_ok.append(sum(w in result["answer"].lower() for w in words) / len(words) >= 0.5)
    n = len(items)
    return {
        "n": n,
        "hit1": sum(r == 0 for r in ranks) / n,
        "hitk": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / (r + 1) for r in ranks if r is not None) / n,
        "keywords": sum(keyword_ok) / n,
        "p50": statistics.median(times),
        "p95": sorted(times)[int(0.95 * (n - 1))],
    }


def table(results, top_k, with_llm):
    names = list(results)
    pct = lambda v: f"{v * 100:.1f}%"
    rows = [
        ("Questions", lambda r: str(r["n"])),
        ("Hit@1", lambda r: pct(r["hit1"])),
        (f"Hit@{top_k}", lambda r: pct(r["hitk"])),
        ("MRR", lambda r: f"{r['mrr']:.3f}"),
        ("Answer accuracy (keywords)" if with_llm else "Context keyword coverage", lambda r: pct(r["keywords"])),
        ("Latency p50 / p95", lambda r: f"{r['p50'] * 1000:.0f} / {r['p95'] * 1000:.0f} ms"),
    ]
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(results[b]) if "error" not in results[b] else "n/a" for b in names) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--eval-set", default=str(Path(__file__).parent / "eval_set.json"))
    ap.add_argument("--backends", nargs="+", default=["tfidf"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    items = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    if any(i["question"].startswith("REPLACE") for i in items):
        sys.exit("eval_set.json still has template entries. Replace them with real questions first.")

    from dotenv import load_dotenv

    load_dotenv()
    if not args.with_llm:
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"  # unreachable -> extractive, no LLM
    sys.path.insert(0, str(ROOT))
    from src.qa_engine import RAGPipeline

    results = {}
    for backend in args.backends:
        try:
            pipeline = RAGPipeline(docs_dir=args.docs, backend=backend, top_k=args.top_k)
            pipeline.build()
            results[backend] = evaluate(pipeline, items)
        except Exception as exc:  # e.g. sentence-transformers not installed
            results[backend] = {"error": str(exc)}
            print(f"[{backend}] skipped: {exc}", file=sys.stderr)

    out = table(results, args.top_k, args.with_llm)
    print("\n" + out + "\n")
    if args.update_readme:
        text = README.read_text(encoding="utf-8")
        block = f"{START}\n{out}\n{END}"
        README.write_text(re.sub(f"{START}.*?{END}", lambda _: block, text, flags=re.S), encoding="utf-8")


if __name__ == "__main__":
    main()
