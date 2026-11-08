"""Text cleaning for text extracted from PDFs."""
import re
import unicodedata

_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


def clean_text(text: str) -> str:
    """Normalise unicode, repair hyphenated line breaks, collapse whitespace.
    Paragraph breaks (blank lines) are preserved."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join "learn-\ning"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # single newline -> space
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
