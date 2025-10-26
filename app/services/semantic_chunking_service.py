"""
Service de chunking sémantique dynamique avec seuil adaptatif.
"""

import numpy as np
from typing import List, Dict, Any
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

        # Configuration du chunking sémantique (min_chunk_size en mots)
        self.base_similarity_threshold = 0.75
        self.min_chunk_size_words = 30    # recommandé : 30 mots minimum par chunk
        self.max_chunk_size_chars = 2000  # limitation en caractères pour éviter des chunks trop gros
        self.adaptive_window = 5

        # cache local des embeddings de segments pour éviter recomputation
        self._segment_embedding_cache: Dict[str, List[float]] = {}

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divise le texte en phrases/paragraphes en préservant la structure.
        """
        sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])',  # considère majuscules accentuées
            r'(?<=[.!?])\s*\n\s*',
            r'\n\s*\n',
        ]

        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = [s.strip() for s in new_sentences if s.strip()]

        return sentences

    def _get_segment_embedding(self, text: str) -> List[float]:
        """
        Récupère l'embedding d'un segment avec cache.
        Utilise la méthode embed_query de LocalEmbeddingService.
        """
        key = text[:512]  # clé courte mais suffisante; on peut hasher
        if key in self._segment_embedding_cache:
            return self._segment_embedding_cache[key]

        emb = self.embedding_service.embed_query(text)
        # s'assurer que l'embedding est une liste/np.array
        emb_arr = np.array(emb, dtype=float).tolist()
        self._segment_embedding_cache[key] = emb_arr
        return emb_arr

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité cosinus entre deux textes (via leurs embeddings).
        """
        try:
            embedding1 = self._get_segment_embedding(text1)
            embedding2 = self._get_segment_embedding(text2)

            arr1 = np.array(embedding1, dtype=float).reshape(1, -1)
            arr2 = np.array(embedding2, dtype=float).reshape(1, -1)

            sim = cosine_similarity(arr1, arr2)[0, 0]
            return float(sim)
        except Exception as e:
            print(f"Erreur lors du calcul de similarité: {e}")
            return 0.0

    def calculate_adaptive_threshold(self, current_chunk: List[str], base_threshold: float) -> float:
        """
        Calcule un seuil adaptatif basé sur la variance locale des similarités.
        """
        if len(current_chunk) < 2:
            return base_threshold

        try:
            similarities = []
            # ne pas recalculer embeddings inutilement : on utilise _get_segment_embedding
            for i in range(len(current_chunk) - 1):
                sim = self.calculate_similarity(current_chunk[i], current_chunk[i + 1])
                similarities.append(sim)

            if not similarities:
                return base_threshold

            mean_similarity = float(np.mean(similarities))
            std_similarity = float(np.std(similarities))

            adaptive_threshold = base_threshold - (std_similarity * 0.1)
            adaptive_threshold = max(0.3, min(0.9, adaptive_threshold))
            return adaptive_threshold
        except Exception as e:
            print(f"Erreur lors du calcul du seuil adaptatif: {e}")
            return base_threshold

    def merge_chunk(self, chunk_segments: List[str]) -> str:
        """
        Fusionne les segments d'un chunk en un texte cohérent.
        """
        merged_text = " ".join(chunk_segments)
        merged_text = re.sub(r'\s+', ' ', merged_text)
        return merged_text.strip()

    def create_semantic_chunks(self, text: str, task_id: str) -> List[Dict[str, Any]]:
        """
        Crée des chunks sémantiques dynamiques.
        """
        try:
            segments = self.split_into_sentences(text)
            if not segments:
                return []

            chunks: List[Dict[str, Any]] = []
            current_chunk = [segments[0]]
            adaptive_threshold = self.base_similarity_threshold

            print(f"Début du chunking sémantique pour {len(segments)} segments...")

            for i in range(1, len(segments)):
                last_segment = current_chunk[-1]
                next_segment = segments[i]

                similarity = self.calculate_similarity(last_segment, next_segment)
                adaptive_threshold = self.calculate_adaptive_threshold(current_chunk, self.base_similarity_threshold)

                chunk_text = self.merge_chunk(current_chunk)
                new_chunk_text = self.merge_chunk(current_chunk + [next_segment])

                # critères (taille en mots et taille en caractères)
                new_chunk_word_count = len(new_chunk_text.split())
                new_chunk_char_count = len(new_chunk_text)

                should_continue = (
                    similarity > adaptive_threshold
                    and new_chunk_char_count <= self.max_chunk_size_chars
                )

                if should_continue:
                    current_chunk.append(next_segment)
                else:
                    # finaliser si le chunk courant a assez de mots
                    chunk_word_count = len(chunk_text.split())
                    if chunk_word_count >= self.min_chunk_size_words:
                        chunk_data = {
                            "content": chunk_text,
                            "metadata": {
                                "task_id": task_id,
                                "chunk_id": f"{task_id}_semantic_chunk_{len(chunks)}",
                                "chunk_index": len(chunks),
                                "chunk_type": "semantic_dynamic",
                                "segment_count": len(current_chunk),
                                "word_count": chunk_word_count,
                                "character_count": len(chunk_text),
                                "similarity_threshold_used": adaptive_threshold,
                                "created_at": datetime.now().isoformat(),
                                "source": "transcript"
                            }
                        }
                        chunks.append(chunk_data)

                    # démarrer nouveau chunk
                    current_chunk = [next_segment]

                if i % 10 == 0:
                    print(f"Traitement segment {i}/{len(segments)} - Chunks créés: {len(chunks)}")

            # ajouter dernier chunk
            last_merged = self.merge_chunk(current_chunk)
            if last_merged:
                last_word_count = len(last_merged.split())
                if last_word_count >= self.min_chunk_size_words:
                    chunk_data = {
                        "content": last_merged,
                        "metadata": {
                            "task_id": task_id,
                            "chunk_id": f"{task_id}_semantic_chunk_{len(chunks)}",
                            "chunk_index": len(chunks),
                            "chunk_type": "semantic_dynamic",
                            "segment_count": len(current_chunk),
                            "word_count": last_word_count,
                            "character_count": len(last_merged),
                            "similarity_threshold_used": adaptive_threshold,
                            "created_at": datetime.now().isoformat(),
                            "source": "transcript"
                        }
                    }
                    chunks.append(chunk_data)

            print(f"Chunking sémantique terminé: {len(chunks)} chunks créés")
            return chunks

        except Exception as e:
            print(f"Erreur lors du chunking sémantique: {e}")
            return []

    def get_chunking_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcule les statistiques du chunking.
        """
        if not chunks:
            return {}

        chunk_sizes = [len(chunk["content"].split()) for chunk in chunks]
        segment_counts = [chunk["metadata"]["segment_count"] for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_words": float(np.mean(chunk_sizes)),
            "min_chunk_size_words": int(np.min(chunk_sizes)),
            "max_chunk_size_words": int(np.max(chunk_sizes)),
            "avg_segments_per_chunk": float(np.mean(segment_counts)),
            "chunking_strategy": "semantic_dynamic",
            "base_similarity_threshold": self.base_similarity_threshold
        }
