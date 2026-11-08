"""The same RAG workflow rebuilt with LangChain (>=1.0 style, LCEL runnables).
Optional: the main app does not depend on it. Requires: langchain-core, langchain-community,
langchain-text-splitters, langchain-chroma, langchain-ollama."""
from __future__ import annotations
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
from .prompt_templates import NOT_FOUND
from .utils import ROOT


class ProjectEmbeddings(Embeddings):
    """Adapter: lets LangChain use our own embedder (MiniLM or hashing)."""
    def __init__(self, embedder):
        self.embedder = embedder
    def embed_documents(self, texts):
        return self.embedder.embed(list(texts)).tolist()
    def embed_query(self, text):
        return self.embedder.embed([text])[0].tolist()


def load_and_split(pdf_path, chunk_size=800, chunk_overlap=120):
    docs = PyPDFLoader(str(pdf_path)).load()          # one Document per page, metadata has 'page' (0-based)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", pdf_path)).name
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1   # make it 1-based like our own pipeline
    return chunks


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


def build_chain(vectorstore: Chroma, llm, top_k: int = 4):
    system = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "<context>\n{context}\n</context>\n\nQuestion: {question}\n\nAnswer (with [n] citations):")])
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())
