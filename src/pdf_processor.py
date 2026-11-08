"""Safe PDF validation and page-by-page text extraction."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader
from .text_cleaner import clean_text
from .utils import get_logger

log = get_logger(__name__)


class PDFValidationError(ValueError):
    """Raised when a file is not an acceptable PDF."""


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


def validate_pdf(path: str | Path, max_mb: float = 25, max_pages: int = 800) -> None:
    p = Path(path)
    if not p.exists():
        raise PDFValidationError(f"File not found: {p.name}")
    if p.suffix.lower() != ".pdf":
        raise PDFValidationError("Only .pdf files are supported.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise PDFValidationError(f"File is {size_mb:.1f} MB; limit is {max_mb} MB.")
    with open(p, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise PDFValidationError("File does not look like a real PDF (bad header).")
    try:
        reader = PdfReader(str(p))
        if reader.is_encrypted:
            raise PDFValidationError("Encrypted PDFs are not supported.")
        if len(reader.pages) > max_pages:
            raise PDFValidationError(f"PDF has {len(reader.pages)} pages; limit is {max_pages}.")
    except PDFValidationError:
        raise
    except Exception as e:  # corrupt file
        raise PDFValidationError(f"Could not read PDF: {e}") from e


def extract_pages(path: str | Path) -> list[PageText]:
    """Return cleaned text for every non-empty page. Scanned (image-only) pages
    yield no text and are skipped; see Chapter 11 for the OCR extension."""
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            log.warning("Page %d of %s failed: %s", i, Path(path).name, e)
            continue
        text = clean_text(raw)
        if text:
            pages.append(PageText(i, text))
        else:
            log.info("Page %d of %s has no extractable text", i, Path(path).name)
    return pages


def extract_metadata(path: str | Path) -> dict:
    reader = PdfReader(str(path))
    meta = reader.metadata or {}
    return {"title": str(meta.get("/Title") or Path(path).stem),
            "author": str(meta.get("/Author") or ""),
            "num_pages": len(reader.pages)}
