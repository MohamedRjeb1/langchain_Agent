"""
Service d'embedding local utilisant Ollama (nomic-embed-text).
Compatible avec LangChain.
"""

from typing import List, Dict, Any
import hashlib
from app.core.config import get_settings
from langchain_ollama import OllamaEmbeddings


class LocalEmbeddingService:
    """
    Service d'embedding local utilisant OllamaEmbeddings.
    Fournit des embeddings sans dépendance à une API distante.
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedding_cache = {}
        self.model_name = "nomic-embed-text"
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialise le modèle local Ollama."""
        try:
            self.model = OllamaEmbeddings(model=self.model_name)
            print(f"Modèle d'embedding chargé : {self.model_name}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle local: {str(e)}")
            print("Assurez-vous qu'Ollama est installé et que le modèle est disponible localement.")
            raise e

    def _generate_cache_key(self, text: str) -> str:
        """Génère une clé unique pour le cache."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_query(self, text: str) -> List[float]:
        """
        Génère un embedding pour une requête unique.
        Utilise le cache si disponible.
        """
        try:
            cache_key = self._generate_cache_key(text)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            embedding = self.model.embed_query(text)

            # Stocker dans le cache
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            print(f"Erreur lors de la génération d'embedding local: {str(e)}")
            # Retourne un vecteur nul si erreur
            return [0.0] * self.get_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour plusieurs documents.
        """
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            print(f"Erreur lors de la génération d'embeddings multiples: {str(e)}")
            return [[0.0] * self.get_embedding_dimension()] * len(texts)

    def get_embedding_dimension(self) -> int:
        """
        Retourne la dimension typique du modèle utilisé.
        (nomic-embed-text a une dimension de 768)
        """
        return 768

    def clear_cache(self):
        """Efface le cache."""
        self.embedding_cache.clear()
        print("Cache des embeddings effacé.")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self.embedding_cache),
            "embedding_dimension": self.get_embedding_dimension(),
            "model_name": self.model_name
        }
