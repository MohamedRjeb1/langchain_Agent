"""Simple LLM service wrapper to centralize LLM calls (Ollama/Mistral).

This module provides a minimal `LLMService` class with a `generate` method.
It makes the ollama import optional so the code can be tested without the runtime.
"""
from typing import Optional, Dict, Any

try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None


class LLMService:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.llm = None

    def _ensure_model(self):
        if self.llm is None:
            if OllamaLLM is None:
                raise RuntimeError("OllamaLLM not available in this environment")
            self.llm = OllamaLLM(model=self.model_name)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Generate text from a prompt and return structured output.

        Returns a dict with keys: text (str) and raw (model raw output if any).
        """
        self._ensure_model()
        # Use the underlying API; keep wrapper minimal
        out = self.llm.invoke(prompt)
        return {"text": out, "raw": out}
