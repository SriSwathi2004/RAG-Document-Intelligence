"""Turn a PDF file into chunks ready for embedding."""
from __future__ import annotations
from pathlib import Path
from .chunking import chunk_pages, Chunk
from .pdf_processor import validate_pdf, extract_pages, extract_metadata
from .utils import file_sha256


def load_document(path: str | Path, cfg: dict) -> tuple[dict, list[Chunk]]:
    """Validate, extract, clean and chunk one PDF. Returns (doc_info, chunks).
    Raises PDFValidationError for bad files and ValueError if no text is found."""
    path = Path(path)
    validate_pdf(path, cfg["security"]["max_file_mb"], cfg["security"]["max_pages"])
    pages = extract_pages(path)
    if not pages:
        raise ValueError(f"No extractable text in {path.name} (scanned PDF? OCR is needed).")
    doc_id = file_sha256(path)[:16]
    ch = cfg["chunking"]
    chunks = chunk_pages(pages, doc_id, path.name, ch["strategy"], ch["chunk_size"], ch["chunk_overlap"])
    info = {"doc_id": doc_id, "source": path.name, "num_chunks": len(chunks),
            "pages_with_text": len(pages), **extract_metadata(path)}
    return info, chunks
