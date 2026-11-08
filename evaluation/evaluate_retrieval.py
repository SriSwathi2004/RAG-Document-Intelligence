"""Measure retrieval quality on the evaluation dataset (Hit@K, Precision@K, MRR).
A hit means a retrieved chunk matches BOTH the expected source file and page.
Usage: python -m evaluation.evaluate_retrieval --backend hashing --chunk-size 800 --top-k 4"""
import argparse, json, tempfile
from pathlib import Path
from src.embeddings import HashingEmbedder, get_embedder
from src.llm import ExtractiveLLM
from src.rag_pipeline import RAGPipeline
from src.utils import load_config, ROOT


def evaluate(backend="hashing", strategy="recursive", chunk_size=800, overlap=120, top_k=4, min_score=0.0) -> dict:
    cfg = load_config()
    cfg["embeddings"]["backend"] = backend
    cfg["chunking"].update(strategy=strategy, chunk_size=chunk_size, chunk_overlap=overlap)
    cfg["retrieval"].update(top_k=top_k, min_score=min_score)
    data = json.loads((ROOT / "evaluation" / "evaluation_dataset.json").read_text(encoding="utf-8"))
    embedder = HashingEmbedder() if backend == "hashing" else get_embedder(cfg)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = RAGPipeline(cfg, embedder=embedder, llm=ExtractiveLLM(), persist_dir=Path(tmp) / "db")
        for pdf in sorted((ROOT / "data" / "sample_pdfs").glob("*.pdf")):
            p.ingest(pdf)
        hits_at_k = prec = rr = 0.0
        for item in data:
            hits = p.retriever.retrieve(item["question"])
            rel = [h["source"] == item["source"] and h["page"] == item["page"] for h in hits]
            hits_at_k += any(rel)
            prec += (sum(rel) / len(rel)) if rel else 0
            rr += next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
    n = len(data)
    return {"embedder": embedder.name, "strategy": strategy, "chunk_size": chunk_size, "top_k": top_k,
            "questions": n, f"hit@{top_k}": round(hits_at_k / n, 3), f"precision@{top_k}": round(prec / n, 3),
            "MRR": round(rr / n, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hashing"); ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--chunk-size", type=int, default=800); ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.backend, a.strategy, a.chunk_size, a.overlap, a.top_k), indent=2))
