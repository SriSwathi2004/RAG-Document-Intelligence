"""Compare chunk sizes / strategies and write a markdown table (real measured numbers only)."""
import sys
from evaluation.evaluate_retrieval import evaluate

backend = sys.argv[1] if len(sys.argv) > 1 else "hashing"
rows = []
for strategy in ("fixed", "sentence", "recursive"):
    for size in (200, 400, 800):
        for k in (1, 4):
            r = evaluate(backend, strategy, size, size // 6, k)
            rows.append(f"| {strategy} | {size} | {k} | {r[f'hit@{k}']} | {r[f'precision@{k}']} | {r['MRR']} |")
print(f"Embedder: {backend}\n\n| strategy | chunk_size | top_k | hit@k | precision@k | MRR |\n|---|---|---|---|---|---|")
print("\n".join(rows))
