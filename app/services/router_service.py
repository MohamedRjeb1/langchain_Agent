"""LLM-based router that decides whether to use RAG or normal chat.

It asks the LLM to respond ONLY with 'RAG' or 'NORMAL' based on the user message.
"""
from typing import Dict

from app.services.llm_service import LLMService


class LLMBasedRouter:
    def __init__(self, llm: LLMService):
        self.llm = llm
        self.prompt_template = (
            "Tu es un détecteur de type de question.\n"
            "Voici le message de l'utilisateur : \"{user_message}\"\n"
            "Réponds uniquement par 'RAG' si la question demande une information externe,\n"
            "ou 'NORMAL' si c'est une question simple de conversation.\n"
        )

    def detect_mode(self, user_message: str) -> Dict[str, str]:
        """Return {mode: 'rag'|'chat', raw: original_model_text}.

        We normalize the model text to robustly parse 'RAG' or 'NORMAL'.
        """
        if not user_message:
            return {"mode": "chat", "raw": ""}
        prompt = self.prompt_template.format(user_message=user_message)
        out = self.llm.generate(prompt, max_tokens=8, temperature=0.0)
        text = (out.get("text") or "").strip().strip('`').strip()
        upper = text.upper()
        if "RAG" in upper and "NORMAL" not in upper:
            return {"mode": "rag", "raw": text}
        if "NORMAL" in upper and "RAG" not in upper:
            return {"mode": "chat", "raw": text}
        # Fallback: try first token
        tok = upper.split()[0] if upper else ""
        if tok == "RAG":
            return {"mode": "rag", "raw": text}
        return {"mode": "chat", "raw": text}
