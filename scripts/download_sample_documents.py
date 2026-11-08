"""Optional: download extra openly licensed PDFs. Add entries to SOURCES ONLY after verifying the license.
The ZIP already contains self-generated PDFs, so this script is not required to run the project."""
import sys, requests
from pathlib import Path

SOURCES: list[dict] = [
    # {"name": "example.pdf", "url": "https://...", "license": "CC BY 4.0", "attribution": "Author, Title"},
]
out = Path(__file__).resolve().parent.parent / "data" / "sample_pdfs"
if not SOURCES:
    print("No external documents configured. Use the bundled samples (see data/DOCUMENT_SOURCES.md).")
    sys.exit(0)
for s in SOURCES:
    r = requests.get(s["url"], timeout=60); r.raise_for_status()
    (out / s["name"]).write_bytes(r.content); print("downloaded", s["name"], "-", s["license"])
