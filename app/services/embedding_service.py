"""
Service d'embedding local pour éviter les quotas API.
"""
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import hashlib

from app.core.config import get_settings


class LocalEmbeddingService:
    """
    Service d'embedding local utilisant SentenceTransformers.
    Évite les quotas API et fonctionne hors ligne.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.embedding_cache = {}
        
        # Initialiser le modèle local
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle SentenceTransformers."""
        try:
            # Modèle français léger
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print(" Modèle d'embedding local chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle local: {str(e)}")
            print("Installez sentence-transformers avec: pip install sentence-transformers")
            raise e
    
    def _generate_cache_key(self, text: str) -> str:
        """Génère une clé de cache pour le texte."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Génère un embedding pour une requête.
        
        Args:
            text: Texte à embedder
            
        Returns:
            Vecteur d'embedding
        """
        try:
            # Vérifier le cache
            cache_key = self._generate_cache_key(text)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Générer l'embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convertir en liste de flottants
            embedding_list = embedding.tolist()
            
            # Mettre en cache
            self.embedding_cache[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            print(f"Erreur lors de la génération d'embedding local: {str(e)}")
            # Retourner un vecteur zéro en cas d'erreur
            return [0.0] * 384  # Dimension du modèle MiniLM
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour plusieurs documents.
        
        Args:
            texts: Liste des textes à embedder
            
        Returns:
            Liste des vecteurs d'embedding
        """
        try:
            embeddings = []
            
            for text in texts:
                embedding = self.embed_query(text)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            print(f"Erreur lors de la génération d'embeddings multiples: {str(e)}")
            # Retourner des vecteurs zéro
            return [[0.0] * 384] * len(texts)
    
    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        return 384  # Dimension du modèle paraphrase-multilingual-MiniLM-L12-v2
    
    def clear_cache(self):
        """Efface le cache des embeddings."""
        self.embedding_cache.clear()
        print("Cache des embeddings local effacé")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self.embedding_cache),
            "embedding_dimension": self.get_embedding_dimension(),
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2"
        }
