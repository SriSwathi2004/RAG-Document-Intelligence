# Architecture

```
PDF upload -> validate (ext, magic bytes, size, pages, encryption)
  -> pypdf extract per page -> clean_text -> chunk (fixed|sentence|recursive, per page)
  -> metadata {doc_id=sha256[:16], source, page, chunk_index}
  -> embed (MiniLM or hashing) -> ChromaDB (persistent, cosine)

Question -> length check -> embed query -> top-k search (optional doc filter)
  -> score threshold (below = "not found", LLM never called) -> optional MMR
  -> build_prompt (numbered, fenced <context>) -> LLM (Ollama | extractive fallback)
  -> answer + citations (source, page, score, snippet) -> Streamlit UI
```

| Module | Responsibility |
|---|---|
| pdf_processor.py | validation, page text, metadata |
| text_cleaner.py | unicode/ligature/hyphen/whitespace repair |
| chunking.py | 3 strategies, page-preserving |
| embeddings.py | MiniLM backend + hashing fallback |
| vector_store.py | Chroma add/query/list/delete, dedupe by doc_id |
| retriever.py | threshold + MMR |
| prompt_templates.py, llm.py | prompts, Ollama client, extractive fallback |
| rag_pipeline.py | orchestration, input validation, fallback on LLM failure |

Known limits: no OCR (scanned PDFs are rejected with a clear message), tables are flattened to text,
the score threshold must be re-tuned per embedding model, prompt-injection is reduced not eliminated.
