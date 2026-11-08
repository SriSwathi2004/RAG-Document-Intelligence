"""Chunking strategies. Every chunk keeps its source document and page number."""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str
    page: int
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def fixed_size_split(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    out, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
        start += size - overlap
    return out


def sentence_split(text: str, size: int, overlap: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail.split(" ", 1)[-1] + " " + s).strip() if tail else s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def recursive_split(text: str, size: int, overlap: int,
                    separators: tuple[str, ...] = ("\n\n", ". ", " ")) -> list[str]:
    """Split on the largest separator that keeps pieces under `size`, then merge."""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    for sep in separators:
        if sep in text:
            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                break
    else:
        return fixed_size_split(text, size, overlap)
    chunks, cur = [], ""
    for p in parts:
        p = p if sep == "\n\n" else p + (sep if sep != " " else " ")
        if len(p) > size:
            if cur.strip():
                chunks.append(cur.strip()); cur = ""
            chunks.extend(recursive_split(p, size, overlap, separators[1:] or (" ",)))
        elif len(cur) + len(p) > size:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] if overlap else "") + p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


_STRATEGIES = {"fixed": fixed_size_split, "sentence": sentence_split, "recursive": recursive_split}


def chunk_pages(pages, doc_id: str, source: str, strategy: str = "recursive",
                chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    """Chunk page by page so each chunk maps to exactly one page number."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(_STRATEGIES)}")
    fn = _STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for page in pages:
        for piece in fn(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(piece, doc_id, source, page.page_number, len(chunks)))
    return chunks
