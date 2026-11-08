"""Shared helpers: config loading, logging, hashing."""
from __future__ import annotations
import hashlib
import logging
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml; env vars EMBEDDING_BACKEND / LLM_PROVIDER / OLLAMA_MODEL override it."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.getenv("EMBEDDING_BACKEND"):
        cfg["embeddings"]["backend"] = os.environ["EMBEDDING_BACKEND"]
    if os.getenv("LLM_PROVIDER"):
        cfg["llm"]["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("OLLAMA_MODEL"):
        cfg["llm"]["model"] = os.environ["OLLAMA_MODEL"]
    return cfg


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
