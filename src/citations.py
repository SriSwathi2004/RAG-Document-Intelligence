"""Format source citations from retrieved hits."""

def format_sources(hits: list[dict]) -> list[dict]:
    return [{"n": i, "source": h["source"], "page": h["page"], "score": h["score"],
             "snippet": h["text"][:300] + ("..." if len(h["text"]) > 300 else "")}
            for i, h in enumerate(hits, start=1)]
