"""Embedding backends.

- sentence-transformers: real semantic embeddings (needs internet once, to download the model).
- hashing: dependency-free lexical fallback (hashed unigrams+bigrams). It is NOT semantic:
  it matches shared words. Used for offline tests and as an emergency fallback.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np
from .utils import get_logger

log = get_logger(__name__)
_STOP = set("a an the of in on at to is are was were be been and or for with as by it this that "
            "what which who how does do did from into can about its their".split())


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.min_score = 0.10  # lexical cosine scores are much lower than semantic ones

    def _vec(self, text: str) -> np.ndarray:
        toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.min_score = None  # use config.yaml value

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def get_embedder(cfg: dict):
    backend = cfg["embeddings"]["backend"]
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(cfg["embeddings"]["model_name"])
    except Exception as e:
        log.warning("sentence-transformers unavailable (%s). Falling back to hashing embedder.", e)
        return HashingEmbedder()
