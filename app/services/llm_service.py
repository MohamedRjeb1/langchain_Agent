"""Simple LLM service wrapper to centralize LLM calls (Ollama/Mistral).

This module provides a minimal `LLMService` class with a `generate` method.
It prefers using langchain_ollama if available, and falls back to Ollama's HTTP API
to avoid issues with langchain package mismatches (e.g., `module 'langchain' has no attribute 'verbose'`).
"""
from typing import Optional, Dict, Any
import json
import urllib.request
import urllib.error

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
        # Try langchain_ollama first, then fallback to HTTP API if it fails
        try:
            self._ensure_model()
            out = self.llm.invoke(prompt)
            return {"text": out, "raw": out}
        except Exception:
            # Fallback to Ollama HTTP API
            return self._generate_via_http(prompt, max_tokens=max_tokens, temperature=temperature)

    def _generate_via_http(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
                text = obj.get("response") or obj.get("text") or ""
                return {"text": text, "raw": obj}
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            return {"text": "", "raw": {"error": err_body}, "error": f"HTTPError: {e.code}"}
        except Exception as e:
            return {"text": "", "raw": {"error": str(e)}, "error": str(e)}
