def test_top_hit_has_correct_source_and_page(pipeline):
    hits = pipeline.retriever.retrieve("What is the default rate limit of the Notely API?")
    assert hits and hits[0]["source"] == "notely_api_guide.pdf" and hits[0]["page"] == 1


def test_doc_filter(pipeline):
    doc = next(d for d in pipeline.store.list_documents() if d["source"] == "ml_basics.pdf")
    hits = pipeline.retriever.retrieve("gradient descent", [doc["doc_id"]])
    assert hits and all(h["source"] == "ml_basics.pdf" for h in hits)


def test_ingest_is_idempotent(pipeline):
    from conftest import SAMPLES
    info = pipeline.ingest(SAMPLES / "ml_basics.pdf")
    assert info["already_indexed"] is True


def test_remove_document(pipeline):
    doc = pipeline.store.list_documents()[0]
    pipeline.remove_document(doc["doc_id"])
    assert doc["doc_id"] not in [d["doc_id"] for d in pipeline.store.list_documents()]
