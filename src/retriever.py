"""Retrieval with a score threshold and optional Maximal Marginal Relevance (MMR)."""
from __future__ import annotations
import numpy as np


def mmr(query_score_items: list[dict], k: int, lam: float = 0.6) -> list[dict]:
    """Pick k items balancing relevance (score) and diversity (cosine to already chosen)."""
    items = list(query_score_items)
    chosen: list[dict] = []
    while items and len(chosen) < k:
        best, best_val = None, -1e9
        for it in items:
            sim = max((float(np.dot(it["embedding"], c["embedding"])) for c in chosen), default=0.0)
            val = lam * it["score"] - (1 - lam) * sim
            if val > best_val:
                best, best_val = it, val
        chosen.append(best); items.remove(best)
    return chosen


class Retriever:
    def __init__(self, store, top_k: int = 4, min_score: float = 0.25, use_mmr: bool = False):
        self.store, self.top_k, self.min_score, self.use_mmr = store, top_k, min_score, use_mmr

    def retrieve(self, question: str, doc_ids: list[str] | None = None) -> list[dict]:
        fetch = self.top_k * 3 if self.use_mmr else self.top_k
        hits = self.store.query(question, fetch, doc_ids)
        hits = [h for h in hits if h["score"] >= self.min_score]
        if self.use_mmr:
            hits = mmr(hits, self.top_k)
        for h in hits:
            h.pop("embedding", None)
        return hits
