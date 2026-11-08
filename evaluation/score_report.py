"""Print the top similarity score for every evaluation question, to choose retrieval.min_score by measurement.
Usage: python -m evaluation.score_report --backend sentence-transformers"""
import argparse, json, tempfile
from pathlib import Path
from src.embeddings import HashingEmbedder, get_embedder
from src.llm import ExtractiveLLM
from src.rag_pipeline import RAGPipeline
from src.utils import load_config, ROOT


def main(backend: str) -> None:
    cfg = load_config(); cfg["embeddings"]["backend"] = backend
    embedder = HashingEmbedder() if backend == "hashing" else get_embedder(cfg)
    data = json.loads((ROOT / "evaluation" / "evaluation_dataset.json").read_text(encoding="utf-8"))
    unsupported = json.loads((ROOT / "data" / "evaluation" / "unsupported_questions.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = RAGPipeline(cfg, embedder=embedder, llm=ExtractiveLLM(), persist_dir=Path(tmp) / "db")
        for pdf in sorted((ROOT / "data" / "sample_pdfs").glob("*.pdf")):
            p.ingest(pdf)
        print(f"Embedder: {embedder.name}\n\nANSWERABLE questions (top score, and whether the top chunk has the right file+page)")
        good = []
        for q in data:
            top = p.store.query(q["question"], 1)[0]
            ok = top["source"] == q["source"] and top["page"] == q["page"]
            good.append(top["score"])
            print(f"  {top['score']:.3f}  {'OK  ' if ok else 'MISS'}  {q['question']}")
        print("\nUNANSWERABLE questions (top score of the best, but irrelevant, chunk)")
        bad = []
        for q in unsupported:
            top = p.store.query(q, 1)[0]; bad.append(top["score"])
            print(f"  {top['score']:.3f}        {q}")
    lo, hi = min(good), max(bad)
    print(f"\nlowest answerable top score : {lo:.3f}\nhighest unanswerable score  : {hi:.3f}")
    if lo > hi:
        print(f"A threshold between {hi:.3f} and {lo:.3f} separates them on this small set (for example {(lo + hi) / 2:.2f}).")
    else:
        print("The two groups overlap: no single threshold separates them on this set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="sentence-transformers")
    main(ap.parse_args().backend)
