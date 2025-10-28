"""
Service de chunking sémantique dynamique avec seuil adaptatif.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.services.embedding_service import LocalEmbeddingService


class SemanticChunkingService:
    """
    Service de chunking sémantique dynamique qui :
    - Segmente le texte en phrases/paragraphes
    - Calcule la similarité cosinus entre segments
    - Crée des chunks basés sur la cohérence sémantique
    - Utilise un seuil adaptatif pour la similarité
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = LocalEmbeddingService()
        
        # Configuration du chunking sémantique
        self.base_similarity_threshold = 0.65  # Seuil de base
        self.min_chunk_size = 100  # Taille minimale d'un chunk
        self.max_chunk_size = 2000  # Taille maximale d'un chunk
        self.adaptive_window = 5  # Fenêtre pour calculer le seuil adaptatif
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divise le texte en phrases en préservant la structure.
        
        Args:
            text: Texte à segmenter
            
        Returns:
            Liste des phrases
        """
        # Patterns pour diviser en phrases
        sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Après ponctuation, avant majuscule
            r'(?<=[.!?])\s*\n\s*',      # Après ponctuation, nouvelle ligne
            r'\n\s*\n',                 # Double nouvelle ligne (paragraphes)
        ]
        
        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = [s.strip() for s in new_sentences if s.strip()]
        
        return sentences
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité cosinus entre deux textes.
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
            
        Returns:
            Score de similarité cosinus
        """
        try:
            # Générer les embeddings via le service (interface publique)
            if hasattr(self.embedding_service, "embed_query"):
                embedding1 = self.embedding_service.embed_query(text1)
                embedding2 = self.embedding_service.embed_query(text2)
            else:
                # Compatibilité défensive: certains services exposent .embeddings.embed_query
                embedding1 = self.embedding_service.embeddings.embed_query(text1)  # type: ignore[attr-defined]
                embedding2 = self.embedding_service.embeddings.embed_query(text2)  # type: ignore[attr-defined]
            
            # Calculer la similarité cosinus
            similarity = cosine_similarity(
                [embedding1], 
                [embedding2]
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"Erreur lors du calcul de similarité: {str(e)}")
            return 0.0
    
    def calculate_adaptive_threshold(self, current_chunk: List[str], base_threshold: float) -> float:
        """
        Calcule un seuil adaptatif basé sur la variance locale des similarités.
        
        Args:
            current_chunk: Chunk actuel en construction
            base_threshold: Seuil de base
            
        Returns:
            Seuil adaptatif calculé
        """
        if len(current_chunk) < 2:
            return base_threshold
        
        try:
            # Calculer les similarités entre segments consécutifs
            similarities = []
            for i in range(len(current_chunk) - 1):
                sim = self.calculate_similarity(current_chunk[i], current_chunk[i + 1])
                similarities.append(sim)
            
            if not similarities:
                return base_threshold
            
            # Calculer la moyenne et l'écart-type
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Ajuster le seuil basé sur la variance locale
            # Plus la variance est élevée, plus on est strict
            adaptive_threshold = base_threshold - (std_similarity * 0.1)
            
            # S'assurer que le seuil reste dans des limites raisonnables
            adaptive_threshold = max(0.3, min(0.9, adaptive_threshold))
            
            return adaptive_threshold
            
        except Exception as e:
            print(f"Erreur lors du calcul du seuil adaptatif: {str(e)}")
            return base_threshold
    
    def merge_chunk(self, chunk_segments: List[str]) -> str:
        """
        Fusionne les segments d'un chunk en un texte cohérent.
        
        Args:
            chunk_segments: Segments du chunk
            
        Returns:
            Texte fusionné
        """
        # Fusionner avec des espaces appropriés
        merged_text = " ".join(chunk_segments)
        
        # Nettoyer les espaces multiples
        merged_text = re.sub(r'\s+', ' ', merged_text)
        
        return merged_text.strip()
    
    def create_semantic_chunks(self, text: str, task_id: str) -> List[Dict[str, Any]]:
        """
        Crée des chunks sémantiques dynamiques.
        
        Args:
            text: Texte à chunker
            task_id: Identifiant de la tâche
            
        Returns:
            Liste des chunks avec métadonnées
        """
        try:
            # Diviser en phrases/paragraphes
            segments = self.split_into_sentences(text)
            
            if not segments:
                return []
            
            chunks = []
            current_chunk = [segments[0]]
            
            print(f"Début du chunking sémantique pour {len(segments)} segments directs...")
            
            for i in range(1, len(segments)):
                # Calculer la similarité avec le dernier segment du chunk actuel
                similarity = self.calculate_similarity(
                    current_chunk[-1], 
                    segments[i]
                )
                
                # Calculer le seuil adaptatif
                adaptive_threshold = self.calculate_adaptive_threshold(
                    current_chunk, 
                    self.base_similarity_threshold
                )
                
                # Vérifier si on continue le chunk ou on en commence un nouveau
                chunk_text = self.merge_chunk(current_chunk)
                new_chunk_text = self.merge_chunk(current_chunk + [segments[i]])
                
                # Critères pour continuer le chunk :
                # 1. Similarité élevée
                # 2. Taille raisonnable
                # 3. Cohérence sémantique
                should_continue = (
                    similarity > adaptive_threshold and
                    len(new_chunk_text) <= self.max_chunk_size
                )
                
                if should_continue:
                    current_chunk.append(segments[i])
                else:
                    # Finaliser le chunk actuel s'il est assez grand
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_data = {
                            "content": chunk_text,
                            "metadata": {
                                "task_id": task_id,
                                "chunk_id": f"{task_id}_semantic_chunk_{len(chunks)}",
                                "chunk_index": len(chunks),
                                "chunk_type": "semantic_dynamic",
                                "segment_count": len(current_chunk),
                                "word_count": len(chunk_text.split()),
                                "character_count": len(chunk_text),
                                "similarity_threshold_used": adaptive_threshold,
                                "created_at": datetime.now().isoformat(),
                                "source": "transcript"
                            }
                        }
                        chunks.append(chunk_data)
                    
                    # Commencer un nouveau chunk
                    current_chunk = [segments[i]]
                
                # Progress indicator
                if i % 10 == 0:
                    print(f"Traitement segment {i}/{len(segments)} - Chunks créés: {len(chunks)}")
            
            # Ajouter le dernier chunk s'il n'est pas vide
            if current_chunk and len(self.merge_chunk(current_chunk)) >= self.min_chunk_size:
                chunk_text = self.merge_chunk(current_chunk)
                chunk_data = {
                    "content": chunk_text,
                    "metadata": {
                        "task_id": task_id,
                        "chunk_id": f"{task_id}_semantic_chunk_{len(chunks)}",
                        "chunk_index": len(chunks),
                        "chunk_type": "semantic_dynamic",
                        "segment_count": len(current_chunk),
                        "word_count": len(chunk_text.split()),
                        "character_count": len(chunk_text),
                        "similarity_threshold_used": adaptive_threshold,
                        "created_at": datetime.now().isoformat(),
                        "source": "transcript"
                    }
                }
                chunks.append(chunk_data)
            
            print(f"Chunking sémantique terminé: {len(chunks)} chunks créés")
            return chunks
            
        except Exception as e:
            print(f"Erreur lors du chunking sémantique: {str(e)}")
            return []
    
    def get_chunking_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les statistiques du chunking.
        
        Args:
            chunks: Liste des chunks
            
        Returns:
            Statistiques du chunking
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk["content"].split()) for chunk in chunks]
        segment_counts = [chunk["metadata"]["segment_count"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_words": np.mean(chunk_sizes),
            "min_chunk_size_words": np.min(chunk_sizes),
            "max_chunk_size_words": np.max(chunk_sizes),
            "avg_segments_per_chunk": np.mean(segment_counts),
            "chunking_strategy": "semantic_dynamic",
            "base_similarity_threshold": self.base_similarity_threshold
        }
