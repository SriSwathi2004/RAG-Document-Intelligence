"""LLM back-ends. OllamaLLM talks to a local Ollama server; ExtractiveLLM needs no model at all."""
from __future__ import annotations
import re
import requests
from .prompt_templates import NOT_FOUND
from .utils import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.1, timeout: int = 120):
        self.model, self.base_url, self.temperature, self.timeout = model, base_url.rstrip("/"), temperature, timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.ok and any(m["name"].startswith(self.model.split(":")[0]) for m in r.json().get("models", []))
        except requests.RequestException:
            return False

    def generate(self, system: str, prompt: str) -> str:
        try:
            r = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "stream": False,
                "options": {"temperature": self.temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (requests.RequestException, KeyError, ValueError) as e:
            raise LLMError(f"Ollama call failed: {e}") from e


class ExtractiveLLM:
    """No generation: returns the most relevant sentences from the top chunks.
    Useful for offline testing and as a graceful fallback when Ollama is down."""
    def is_available(self) -> bool:
        return True

    def generate(self, system: str, prompt: str) -> str:
        m = re.search(r"<context>\n(.*?)\n</context>", prompt, re.S)
        q = re.search(r"Question: (.*?)\n", prompt)
        if not m:
            return NOT_FOUND
        qwords = set(re.findall(r"[a-z]{4,}", (q.group(1) if q else "").lower()))
        best = []
        for block in m.group(1).split("\n\n["):
            idx = re.match(r"\[?(\d+)\]", block.strip())
            body = block.split("\n", 1)[-1]
            for s in re.split(r"(?<=[.!?])\s+", body):
                overlap = len(qwords & set(re.findall(r"[a-z]{4,}", s.lower())))
                if overlap:
                    best.append((overlap, s.strip(), idx.group(1) if idx else "1"))
        best.sort(key=lambda t: -t[0])
        if not best:
            return NOT_FOUND
        return " ".join(f"{s} [{i}]" for _, s, i in best[:2])


def get_llm(cfg: dict):
    import os
    llm = cfg["llm"]
    if llm["provider"] == "extractive":
        return ExtractiveLLM()
    return OllamaLLM(llm["model"], os.getenv("OLLAMA_URL", "http://localhost:11434"),
                     llm["temperature"], llm["timeout_s"])
