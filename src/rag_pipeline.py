"""End-to-end RAG pipeline: ingest PDFs, retrieve, build prompt, generate, cite."""
from __future__ import annotations
from pathlib import Path
from .citations import format_sources
from .document_loader import load_document
from .embeddings import get_embedder
from .llm import get_llm, ExtractiveLLM, LLMError
from .prompt_templates import build_prompt, load_system_prompt, NOT_FOUND
from .retriever import Retriever
from .utils import load_config, resolve, get_logger
from .vector_store import VectorStore

log = get_logger(__name__)
MAX_QUESTION_CHARS = 1000


class RAGPipeline:
    def __init__(self, cfg: dict | None = None, embedder=None, llm=None, persist_dir=None):
        self.cfg = cfg or load_config()
        self.embedder = embedder or get_embedder(self.cfg)
        self.store = VectorStore(persist_dir or resolve(self.cfg["paths"]["vector_db"]), self.embedder)
        r = self.cfg["retrieval"]
        min_score = getattr(self.embedder, "min_score", None) or r["min_score"]
        self.retriever = Retriever(self.store, r["top_k"], min_score, r["use_mmr"])
        self.llm = llm or get_llm(self.cfg)
        self.system_prompt = load_system_prompt()

    def ingest(self, pdf_path: str | Path) -> dict:
        info, chunks = load_document(pdf_path, self.cfg)
        info["added_chunks"] = self.store.add_chunks(chunks)
        info["already_indexed"] = info["added_chunks"] == 0
        return info

    def ask(self, question: str, doc_ids: list[str] | None = None) -> dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question is empty.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
        hits = self.retriever.retrieve(question, doc_ids)
        if not hits:  # no evidence above threshold: do not call the LLM at all
            return {"answer": NOT_FOUND, "sources": [], "supported": False}
        prompt = build_prompt(question, hits)
        try:
            answer = self.llm.generate(self.system_prompt, prompt)
        except LLMError as e:
            log.warning("LLM failed (%s); using extractive fallback", e)
            answer = ExtractiveLLM().generate(self.system_prompt, prompt)
        supported = NOT_FOUND not in answer
        return {"answer": answer, "sources": format_sources(hits) if supported else [], "supported": supported}

    def remove_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
