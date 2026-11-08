# RAG Document Intelligence

Upload PDFs, ask questions, get answers with source file, page number and retrieved text.
Says "I could not find this in the uploaded documents." when evidence is missing. Fully local, no paid APIs.

## Quick start with Anaconda (Windows, local)
1. Open **Anaconda Prompt** from the Start menu (not Navigator) and go to the extracted folder: `cd C:\path\to\RAG-Document-Intelligence`
2. Create a separate environment (Navigator's "base" should stay clean): `conda create -n rag python=3.11 -y` then `conda activate rag`
3. `pip install -r requirements.txt`
4. In Anaconda Navigator, choose the `rag` environment in the "Applications on" dropdown if you want Jupyter to use it; or run `pip install notebook` inside `rag` and start `jupyter notebook`.
5. `python scripts/initialize_project.py`, then `streamlit run app/streamlit_app.py`

## Quick start (Windows, plain Python venv)
1. Install Python 3.10-3.12.
2. `py -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`  (first run downloads the MiniLM embedding model, about 90 MB)
4. Optional LLM: install Ollama, `ollama pull llama3.2:3b` (use `llama3.2:1b` on low RAM)
5. `python scripts/initialize_project.py` (indexes the bundled sample PDFs)
6. `streamlit run app/streamlit_app.py`
7. Tests: `python -m pytest -q`   Evaluation: `python -m evaluation.evaluate_retrieval --backend sentence-transformers`

Offline mode: `set EMBEDDING_BACKEND=hashing` (lexical matching only) and `llm.provider: extractive` in config.yaml.

## Colab
Local-only: Ollama and the Streamlit app. In Colab you can run steps 3, 5, 7 (`!pip install -r requirements.txt`, `!python ...`)
with the extractive provider. Do not expect Ollama to work well there.

## Verified status
- 25 automated tests pass (pytest).
- Evaluated with sentence-transformers/all-MiniLM-L6-v2 on a small self-written set (16 answerable and 3 out-of-scope questions over 5 sample PDFs): correct source page in the top 4 for 16/16 questions (MRR 1.0), correct citation 16/16, out-of-scope questions refused 3/3. Precision@4 is 0.25 because each sample page is a single chunk.
- Limits: the set is small and written by the project author, so these results are a sanity check, not proof of accuracy. Ollama answer generation has not been evaluated.
- Not implemented: OCR, table extraction, reranking, in-app hybrid search, agents.
