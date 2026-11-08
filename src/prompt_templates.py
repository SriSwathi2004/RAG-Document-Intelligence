"""Prompt construction. Retrieved text is fenced and declared untrusted (prompt-injection defence)."""
from pathlib import Path
from .utils import ROOT

NOT_FOUND = "I could not find this in the uploaded documents."


def load_system_prompt() -> str:
    return (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8")


def build_prompt(question: str, hits: list[dict]) -> str:
    passages = "\n\n".join(f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
                           for i, h in enumerate(hits, start=1))
    return f"<context>\n{passages}\n</context>\n\nQuestion: {question}\n\nAnswer (with [n] citations):"
