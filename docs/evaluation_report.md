# Evaluation report

Every number below was measured by running the scripts in `evaluation/`. Nothing is estimated.

Important limits: The dataset is small: 16 answerable questions and 3 out-of-scope questions over 5 short sample PDFs (1 to 4 pages each). The documents and questions were written by the project author, so the questions share wording with the documents and are easy. Treat these results as a sanity check that the pipeline works, not as proof of accuracy on real documents. Answer wording quality (faithfulness) was not measured, and no Ollama LLM evaluation has been run.

## 1. Main results: MiniLM embeddings (sentence-transformers/all-MiniLM-L6-v2)

Measured on 2026-11-07 on the author's Windows machine (Python 3.14.6). Settings: recursive chunking, 800 characters, 120 overlap, top_k 4. A retrieval counts as correct only if a retrieved chunk has the expected file AND page.

| Measure | Result |
|---|---|
| Questions | 16 |
| Hit@4 | 1.0 (16 of 16) |
| MRR | 1.0 |
| Precision@4 | 0.25 |
| Correct source cited in the answer's sources (extractive answers) | 16 of 16 |
| Out-of-scope questions correctly refused | 3 of 3 |

Precision@4 is 0.25 because each sample page is a single chunk, so at most 1 of the 4 retrieved chunks can be the correct one. It is the maximum possible here, not a weakness.

### Similarity score distribution (`python -m evaluation.score_report`)

- Top score for the answerable questions: lowest 0.289 ("How does RAG reduce hallucination?"); most were between 0.5 and 0.88.
- Top score for the 3 out-of-scope questions: 0.067, 0.076 and 0.085.
- The two groups do not overlap. The gap (0.085 to 0.289) is why `retrieval.min_score` is set to a value inside it. Only 3 out-of-scope questions were used, so the margin on that side is lightly tested.

## 2. Offline fallback (hashing embedder, lexical only)

Run in the build sandbox on 2026-11-06 because Hugging Face downloads were unavailable there. It matches shared words only, not meaning.

| Measure | Result |
|---|---|
| Hit@4 | 0.938 |
| Precision@4 | 0.698 |
| MRR | 0.938 |
| Correct citation | 15 of 16 |
| Out-of-scope refused | 3 of 3 |

### Chunking comparison (hashing embedder only)

| strategy | chunk_size | top_k | hit@k | precision@k | MRR |
|---|---|---|---|---|---|
| fixed | 200 | 1 | 0.938 | 0.938 | 0.938 |
| fixed | 200 | 4 | 0.938 | 0.599 | 0.938 |
| fixed | 400 | 1 | 0.812 | 0.812 | 0.812 |
| fixed | 400 | 4 | 0.875 | 0.531 | 0.844 |
| fixed | 800 | 1 | 0.938 | 0.938 | 0.938 |
| fixed | 800 | 4 | 0.938 | 0.698 | 0.938 |
| sentence | 200 | 1 | 0.875 | 0.875 | 0.875 |
| sentence | 200 | 4 | 0.938 | 0.557 | 0.906 |
| sentence | 400 | 1 | 0.938 | 0.938 | 0.938 |
| sentence | 400 | 4 | 0.938 | 0.651 | 0.938 |
| sentence | 800 | 1 | 0.938 | 0.938 | 0.938 |
| sentence | 800 | 4 | 0.938 | 0.698 | 0.938 |
| recursive | 200 | 1 | 0.875 | 0.875 | 0.875 |
| recursive | 200 | 4 | 0.938 | 0.542 | 0.906 |
| recursive | 400 | 1 | 0.938 | 0.938 | 0.938 |
| recursive | 400 | 4 | 0.938 | 0.651 | 0.938 |
| recursive | 800 | 1 | 0.938 | 0.938 | 0.938 |
| recursive | 800 | 4 | 0.938 | 0.698 | 0.938 |

The chunking comparison has not been repeated with MiniLM. Differences between strategies on this tiny corpus are small and should not be generalised.

## 3. Not yet measured

- Chunk-size comparison with MiniLM (`python -m evaluation.compare_configs sentence-transformers`).
- Answer faithfulness with a real LLM (Ollama) and RAGAS or manual review.
- Performance on real, longer documents with independently written questions.

## Reproduce

    python -m evaluation.evaluate_retrieval --backend sentence-transformers
    python -m evaluation.evaluate_answers --backend sentence-transformers --provider extractive
    python -m evaluation.score_report --backend sentence-transformers
    python -m evaluation.compare_configs sentence-transformers
