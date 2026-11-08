import pytest
from src.chunking import fixed_size_split, sentence_split, recursive_split, chunk_pages
from src.pdf_processor import PageText

TEXT = " ".join(f"Sentence number {i} talks about machine learning." for i in range(40))


@pytest.mark.parametrize("fn", [fixed_size_split, sentence_split, recursive_split])
def test_chunks_respect_size(fn):
    chunks = fn(TEXT, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)  # small tolerance for overlap/merge


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        fixed_size_split(TEXT, 100, 100)


def test_chunk_pages_keeps_metadata():
    pages = [PageText(1, TEXT), PageText(3, "short page")]
    chunks = chunk_pages(pages, "d1", "x.pdf", "recursive", 300, 50)
    assert {c.page for c in chunks} == {1, 3}
    assert all(c.source == "x.pdf" and c.doc_id == "d1" for c in chunks)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_pages([], "d", "s", "magic")
