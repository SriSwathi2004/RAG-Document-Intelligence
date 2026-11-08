"""Create folders, generate sample PDFs if missing, and pre-index them."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_sample_pdfs import build
from src.rag_pipeline import RAGPipeline
from src.utils import resolve

pipe = RAGPipeline()
sample_dir = resolve(pipe.cfg["paths"]["sample_pdfs"])
if not list(sample_dir.glob("*.pdf")):
    build(sample_dir)
for pdf in sorted(sample_dir.glob("*.pdf")):
    info = pipe.ingest(pdf)
    print(pdf.name, "-> already indexed" if info["already_indexed"] else f"-> {info['added_chunks']} chunks")
print("Done. Start the app with: streamlit run app/streamlit_app.py")
