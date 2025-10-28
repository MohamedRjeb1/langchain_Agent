"""
Wrapper FAISS pour indexer des embeddings avec similarité cosinus.
- Normalisation L2 obligatoire avant insertion et requête (cosine = inner product après L2).
- Sauvegarde: faiss.write_index + JSON (mapping id -> document/metadata).
- Verrouillage de fichier pour éviter les corruptions concurrentes.
"""
from __future__ import annotations
import os
import json
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None

try:
    from filelock import FileLock
except Exception:  # pragma: no cover
    FileLock = None


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / norms).astype(np.float32)


class FAISSIndex:
    """Index FAISS basé sur Inner Product pour approximer la cosine similarity.

    Convention: toutes les opérations utilisent des vecteurs L2-normalisés.
    Ainsi, score retourné ∈ [-1, 1] correspond au cosinus; on clip ensuite à [0,1].
    """

    def __init__(self, dim: int, index_dir: str, task_id: str):
        if faiss is None:
            raise RuntimeError("faiss n'est pas installé. Installez 'faiss-cpu'.")
        self.dim = dim
        self.index_dir = index_dir
        self.task_id = task_id
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, f"{task_id}.faiss")
        self.map_path = os.path.join(index_dir, f"{task_id}_docs.json")
        self._index = faiss.IndexFlatIP(dim)
        self._doc_map: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._file_lock_path = self.index_path + ".lock"

    # ---------- Persistence ----------
    def save(self) -> None:
        """Sauvegarde l'index FAISS et la map doc sous lock (process + thread)."""
        if FileLock is not None:
            with FileLock(self._file_lock_path + ".flock"):
                self._save_unlocked()
        else:
            with self._lock:
                self._save_unlocked()

    def _save_unlocked(self) -> None:
        faiss.write_index(self._index, self.index_path)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump({str(i): v for i, v in self._doc_map.items()}, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Charge l'index et la map sous lock. Retourne True si réussi."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.map_path):
            return False
        if FileLock is not None:
            with FileLock(self._file_lock_path + ".flock"):
                return self._load_unlocked()
        else:
            with self._lock:
                return self._load_unlocked()

    def _load_unlocked(self) -> bool:
        try:
            self._index = faiss.read_index(self.index_path)
            with open(self.map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # keys are strings in JSON
            self._doc_map = {int(k): v for k, v in data.items()}
            self._next_id = (max(self._doc_map.keys()) + 1) if self._doc_map else 0
            return True
        except Exception:
            return False

    # ---------- Insertions ----------
    def add(self, embeddings: List[List[float]] | np.ndarray, docs: List[Dict[str, Any]]) -> List[int]:
        """Ajoute des embeddings et leur mapping document.
        - embeddings: shape (n, dim)
        - docs: Liste de dicts {"content": str, "metadata": dict}
        Retourne la liste des ids ajoutés.
        """
        embs = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        assert embs.shape[1] == self.dim
        ids = list(range(self._next_id, self._next_id + embs.shape[0]))
        with self._lock:
            self._index.add(embs)
            for i, d in zip(ids, docs):
                self._doc_map[i] = {
                    "content": d.get("content", ""),
                    "metadata": d.get("metadata", {}),
                }
            self._next_id += embs.shape[0]
        return ids

    # ---------- Search ----------
    def search(self, query_emb: List[float] | np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Recherche k plus proches voisins; retourne liste avec similarity ∈ [0,1]."""
        q = l2_normalize(np.asarray(query_emb, dtype=np.float32))
        D, I = self._index.search(q, k)
        D = D[0]
        I = I[0]
        out: List[Dict[str, Any]] = []
        for sim_raw, idx in zip(D, I):
            if idx == -1:
                continue
            # sim_raw est l'inner product (cosine approx). Clip à [0,1]
            sim = float(max(0.0, min(1.0, sim_raw)))
            doc = self._doc_map.get(int(idx))
            if not doc:
                continue
            out.append({
                "id": int(idx),
                "similarity": sim,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
            })
        # Reranking léger (déjà trié par FAISS, mais on s'assure de l'ordre décroissant)
        out.sort(key=lambda x: x["similarity"], reverse=True)
        return out

    def reconstruct(self, idx: int) -> Optional[np.ndarray]:
        """Reconstitue le vecteur stocké (IndexFlatIP le permet)."""
        try:
            v = self._index.reconstruct(idx)
            return np.asarray(v, dtype=np.float32)
        except Exception:
            return None

    # ---------- Accessors ----------
    @property
    def size(self) -> int:
        return self._index.ntotal

    def get_doc(self, idx: int) -> Optional[Dict[str, Any]]:
        return self._doc_map.get(idx)
