import pytest
from pathlib import Path
from src.pdf_processor import validate_pdf, extract_pages, PDFValidationError
from src.text_cleaner import clean_text
from conftest import SAMPLES


def test_extract_pages_keeps_page_numbers():
    pages = extract_pages(SAMPLES / "ml_basics.pdf")
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    assert "supervised learning" in pages[0].text.lower()


def test_clean_text_fixes_hyphen_and_ligature():
    assert clean_text("learn-\ning \ufb01le   x") == "learning file x"


def test_invalid_extension(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("hi")
    with pytest.raises(PDFValidationError):
        validate_pdf(f)


def test_fake_pdf_rejected(tmp_path):
    f = tmp_path / "fake.pdf"; f.write_bytes(b"not a pdf at all")
    with pytest.raises(PDFValidationError):
        validate_pdf(f)


def test_too_large(tmp_path):
    f = tmp_path / "big.pdf"; f.write_bytes(b"%PDF-" + b"0" * 2_000_000)
    with pytest.raises(PDFValidationError):
        validate_pdf(f, max_mb=1)


def test_missing_file():
    with pytest.raises(PDFValidationError):
        validate_pdf(Path("nope.pdf"))
