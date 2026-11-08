"""Streamlit UI. Run from the project root:  streamlit run app/streamlit_app.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
from src.pdf_processor import PDFValidationError
from src.rag_pipeline import RAGPipeline
from src.utils import resolve

st.set_page_config(page_title="Document Intelligence (RAG)", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipe = get_pipeline()
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Process uploads"):
        upload_dir = resolve(pipe.cfg["paths"]["uploads"]); upload_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = upload_dir / Path(f.name).name  # strip any path components
            target.write_bytes(f.getbuffer())
            with st.status(f"Processing {f.name}...", expanded=False) as status:
                try:
                    info = pipe.ingest(target)
                    msg = "already indexed" if info["already_indexed"] else f"{info['added_chunks']} chunks added"
                    status.update(label=f"{f.name}: {msg}", state="complete")
                except (PDFValidationError, ValueError) as e:
                    status.update(label=f"{f.name}: {e}", state="error")
    if st.button("Load bundled sample PDFs"):
        for pdf in sorted(resolve(pipe.cfg["paths"]["sample_pdfs"]).glob("*.pdf")):
            pipe.ingest(pdf)
        st.success("Sample PDFs indexed.")
    docs = pipe.store.list_documents()
    for d in docs:
        c1, c2 = st.columns([4, 1])
        c1.write(f"{d['source']} ({d['chunks']} chunks)")
        if c2.button("X", key=d["doc_id"], help="Remove document"):
            pipe.remove_document(d["doc_id"]); st.rerun()
    options = {d["source"]: d["doc_id"] for d in docs}
    selected = st.multiselect("Search only in", list(options), default=[])
    st.caption(f"Embeddings: {pipe.embedder.name} | LLM: {pipe.cfg['llm']['provider']}")
    if st.button("Clear conversation"):
        st.session_state["history"] = []; st.rerun()

st.title("Ask your documents")
if not docs:
    st.info("Upload a PDF or load the sample PDFs from the sidebar to begin.")

for turn in st.session_state["history"]:
    with st.chat_message("user"): st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"]["answer"])
        for s in turn["a"]["sources"]:
            with st.expander(f"[{s['n']}] {s['source']} - page {s['page']} (similarity {s['score']})"):
                st.write(s["snippet"])

q = st.chat_input("Ask a question about your documents", disabled=not docs)
if q:
    try:
        with st.spinner("Searching and generating..."):
            res = pipe.ask(q, [options[s] for s in selected] or None)
        st.session_state["history"].append({"q": q, "a": res}); st.rerun()
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:  # never show a stack trace to the user
        st.error(f"Something went wrong: {e}")
