import pytest
pytest.importorskip("langchain_core"); pytest.importorskip("langchain_chroma"); pytest.importorskip("langchain_community")
from langchain_core.runnables import RunnableLambda
from src.embeddings import HashingEmbedder
from src.langchain_rag import ProjectEmbeddings, Chroma, load_and_split, build_chain
from conftest import SAMPLES


def test_langchain_chain_receives_numbered_context(tmp_path):
    store = Chroma(collection_name="lc_test", embedding_function=ProjectEmbeddings(HashingEmbedder()),
                   persist_directory=str(tmp_path))
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        store.add_documents(load_and_split(pdf))
    seen = {}
    def fake_llm(prompt_value):
        seen["text"] = prompt_value.to_string(); return "ok"
    chain = build_chain(store, RunnableLambda(fake_llm), top_k=2)
    assert chain.invoke("What is the default rate limit of the Notely API?") == "ok"
    assert "[1] (source: notely_api_guide.pdf, page 1)" in seen["text"]
