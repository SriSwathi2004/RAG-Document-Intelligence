"""Persistent vector store on ChromaDB (cosine space). We pass our own embeddings."""
from __future__ import annotations
from pathlib import Path
import chromadb
from .chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str | Path, embedder, collection: str = "documents"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # Collection name includes the embedder so different models never mix vectors.
        safe = "".join(c if c.isalnum() else "_" for c in embedder.name)[:40]
        self.col = self.client.get_or_create_collection(f"{collection}_{safe}".strip("_"),
                                                        metadata={"hnsw:space": "cosine"})

    def has_document(self, doc_id: str) -> bool:
        return bool(self.col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        if self.has_document(chunks[0].doc_id):
            return 0  # avoid repeated processing
        vecs = self.embedder.embed([c.text for c in chunks])
        self.col.add(
            ids=[f"{c.doc_id}-{c.page}-{c.chunk_index}" for c in chunks],
            embeddings=vecs.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "source": c.source, "page": c.page,
                        "chunk_index": c.chunk_index} for c in chunks])
        return len(chunks)

    def query(self, question: str, top_k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        q = self.embedder.embed([question])[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=min(top_k, n), where=where,
                             include=["documents", "metadatas", "distances", "embeddings"])
        out = []
        for doc, meta, dist, emb in zip(res["documents"][0], res["metadatas"][0],
                                        res["distances"][0], res["embeddings"][0]):
            out.append({"text": doc, **meta, "score": round(1.0 - float(dist), 4), "embedding": emb})
        return out

    def list_documents(self) -> list[dict]:
        data = self.col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in data["metadatas"]:
            d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "source": m["source"], "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})
