import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import load_config
from src.embeddings import HashingEmbedder
from src.llm import ExtractiveLLM
from src.rag_pipeline import RAGPipeline

SAMPLES = ROOT / "data" / "sample_pdfs"


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["embeddings"]["backend"] = "hashing"
    c["llm"]["provider"] = "extractive"
    return c


@pytest.fixture()
def pipeline(cfg, tmp_path):
    p = RAGPipeline(cfg, embedder=HashingEmbedder(), llm=ExtractiveLLM(), persist_dir=tmp_path / "db")
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        p.ingest(pdf)
    return p
