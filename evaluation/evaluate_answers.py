"""Answer-level check: citation correctness and unsupported-question refusal (no LLM judge, no invented scores).
Usage: python -m evaluation.evaluate_answers --backend sentence-transformers --provider ollama"""
import argparse, json, tempfile
from pathlib import Path
from src.embeddings import HashingEmbedder, get_embedder
from src.llm import get_llm
from src.prompt_templates import NOT_FOUND
from src.rag_pipeline import RAGPipeline
from src.utils import load_config, ROOT


def run(backend="hashing", provider="extractive") -> dict:
    cfg = load_config(); cfg["embeddings"]["backend"] = backend; cfg["llm"]["provider"] = provider
    embedder = HashingEmbedder() if backend == "hashing" else get_embedder(cfg)
    data = json.loads((ROOT / "evaluation" / "evaluation_dataset.json").read_text(encoding="utf-8"))
    unsupported = json.loads((ROOT / "data" / "evaluation" / "unsupported_questions.json").read_text())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        p = RAGPipeline(cfg, embedder=embedder, llm=get_llm(cfg), persist_dir=Path(tmp) / "db")
        for pdf in sorted((ROOT / "data" / "sample_pdfs").glob("*.pdf")):
            p.ingest(pdf)
        cited = sum(any(s["source"] == q["source"] and s["page"] == q["page"] for s in p.ask(q["question"])["sources"]) for q in data)
        refused = sum(p.ask(q)["answer"] == NOT_FOUND for q in unsupported)
    return {"embedder": embedder.name, "llm": provider, "answerable_questions": len(data), "correct_citation": cited,
            "unsupported_questions": len(unsupported), "correctly_refused": refused,
            "note": "Answer text quality (faithfulness) still needs manual review or RAGAS; see Chapter 20."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="hashing"); ap.add_argument("--provider", default="extractive")
    a = ap.parse_args(); print(json.dumps(run(a.backend, a.provider), indent=2))
