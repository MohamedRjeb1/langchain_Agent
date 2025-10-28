"""
Advanced indexing service with robust FAISS backend.

Key improvements vs the previous version:
- Replace fragile DocArrayInMemorySearch + private _embeddings by a public FAISS wrapper.
- Standardize cosine similarity via L2 normalization, returning scores in [0,1].
- Persistence via faiss.write_index + JSON mapping (no pickle).
- File locks to avoid concurrent writes corruption.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from app.services.faiss_index import FAISSIndex, l2_normalize
from app.services.embedding_service import LocalEmbeddingService

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus


class AdvancedIndexingService:
    """
    Advanced indexing service with innovative techniques:
    - Multi-vector indexing strategies
    - Semantic clustering and organization
    - Hybrid search capabilities
    - Dynamic index optimization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.vectorstore_dir = self.settings.VECTORSTORE_DIR
        
        # Ensure vectorstore directory exists
        os.makedirs(self.vectorstore_dir, exist_ok=True)

        # Index storage (in-memory)
        self.indices = {}
        self.index_metadata = {}
        # Lazy embedder init to avoid connecting to Ollama during tests or when not needed
        self.embedder = None
    
    def create_semantic_clusters(self, embedding_results: List[Dict[str, Any]], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Create semantic clusters from embeddings using K-means clustering.
        
        Args:
            embedding_results: List of embedding results
            n_clusters: Number of clusters to create
            
        Returns:
            Clustering results with cluster assignments
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Extract embeddings
            embeddings = np.array([result["embedding"] for result in embedding_results])
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            
            # Organize results by cluster (ensure JSON-serializable keys)
            clusters = defaultdict(list)
            for i, (result, label) in enumerate(zip(embedding_results, cluster_labels)):
                key = int(label) if not isinstance(label, (int, str)) else label
                clusters[key].append({
                    "chunk_id": result["document"].metadata["chunk_id"],
                    "content": result["document"].page_content,
                    "metadata": result["document"].metadata,
                    "embedding": result["embedding"]
                })
            
            return {
                # Convert any non-string keys to plain Python int to avoid numpy.int32 keys
                "clusters": {int(k) if not isinstance(k, (int, str)) else k: v for k, v in clusters.items()},
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "silhouette_score": silhouette_avg,
                "n_clusters": n_clusters,
                "total_chunks": len(embedding_results)
            }
            
        except ImportError:
            print("scikit-learn not available, skipping clustering")
            return {"clusters": {}, "error": "scikit-learn not available"}
        except Exception as e:
            return {"clusters": {}, "error": str(e)}
    
    def create_hybrid_index(self, embedding_results: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
        """
        Create hybrid index combining multiple indexing strategies.
        
        Args:
            embedding_results: List of embedding results
            task_id: Task identifier
            
        Returns:
            Index creation result
        """
        try:
            # Prepare docs payload and embeddings
            documents = [res["document"] for res in embedding_results]
            embeddings = [res["embedding"] for res in embedding_results]
            if len(embeddings) == 0:
                raise ValueError("No embeddings provided to build the index")
            dim = len(embeddings[0])
            index = FAISSIndex(dim=dim, index_dir=self.vectorstore_dir, task_id=task_id)
            docs_payload = [
                {"content": d.page_content, "metadata": getattr(d, "metadata", {})}
                for d in documents
            ]
            index.add(embeddings, docs_payload)
            
            # Create semantic clusters
            clustering_result = self.create_semantic_clusters(embedding_results)
            
            # Create index metadata
            index_metadata = {
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "total_chunks": len(embedding_results),
                "embedding_model": self.settings.DEFAULT_EMBEDDING_MODEL,
                "embedding_dimension": dim,
                "chunk_size": self.settings.CHUNK_SIZE,
                "chunk_overlap": self.settings.CHUNK_OVERLAP,
                "clustering": clustering_result,
                "index_type": "faiss"
            }
            
            # Save index & metadata
            index.save()
            index_file = os.path.join(self.vectorstore_dir, f"{task_id}.faiss")
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(index_metadata, f, indent=2, ensure_ascii=False)
            
            # Store in memory
            self.indices[task_id] = index
            self.index_metadata[task_id] = index_metadata
            
            return {
                "task_id": task_id,
                "status": ProcessingStatus.COMPLETED,
                "message": "Hybrid index created successfully",
                "index_file": index_file,
                "metadata_file": metadata_file,
                "total_chunks": len(embedding_results),
                "clusters": len(clustering_result.get("clusters", {})),
                "silhouette_score": clustering_result.get("silhouette_score", 0)
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Index creation failed: {str(e)}",
                "error": str(e)
            }
    
    def load_index(self, task_id: str) -> bool:
        """
        Load index from disk.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.vectorstore_dir, f"{task_id}.faiss")
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            if not os.path.exists(index_path) or not os.path.exists(metadata_file):
                return False

            # Load metadata first to get dimension
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            dim = int(metadata.get("embedding_dimension", self.settings.EMBEDDING_DIMENSION))

            index = FAISSIndex(dim=dim, index_dir=self.vectorstore_dir, task_id=task_id)
            ok = index.load()
            if not ok:
                return False

            # Store in memory
            self.indices[task_id] = index
            self.index_metadata[task_id] = metadata

            return True
            
        except Exception as e:
            print(f"Error loading index for task {task_id}: {str(e)}")
            return False
    
    def search_similar(self, query: str, task_id: str, k: int = 5, similarity_threshold: float = 0.7, query_embedding: Optional[List[float]] = None, embed_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity on FAISS index.
        - Embeddings are L2-normalized at index-time and query-time.
        - Returned similarity is in [0,1].
        
        Args:
            query: Search query
            task_id: Task identifier
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            query_embedding: Optional precomputed embedding for the query
            embed_query: If True and query_embedding is None, embed the query text
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Load index if not in memory
            if task_id not in self.indices:
                if not self.load_index(task_id):
                    return []

            index = self.indices[task_id]

            # Determine query embedding
            qe = query_embedding
            if qe is None and embed_query:
                if self.embedder is None:
                    try:
                        self.embedder = LocalEmbeddingService()
                    except Exception:
                        # If embedder cannot be initialized, we cannot embed the query
                        return []
                qe = self.embedder.embed_query(query)
            if qe is None:
                return []

            # Primary FAISS search (already returns similarity in [0,1])
            results = index.search(qe, k=k)

            # Optional lightweight reranking with reconstructed vectors
            try:
                qe_norm = l2_normalize(np.asarray(qe, dtype=np.float32))[0]
                reranked: List[Dict[str, Any]] = []
                for r in results:
                    rid = r.get("id")
                    rv = index.reconstruct(rid) if rid is not None else None
                    sim = r["similarity"]
                    if rv is not None:
                        rvn = l2_normalize(rv)[0]
                        sim = float(np.dot(qe_norm, rvn))
                        sim = max(0.0, min(1.0, sim))
                    if sim >= similarity_threshold:
                        reranked.append({
                            "content": r["content"],
                            "metadata": r["metadata"],
                            "similarity_score": sim,
                        })
                reranked.sort(key=lambda x: x["similarity_score"], reverse=True)
                return reranked
            except Exception:
                # Fallback: threshold and return as-is
                return [
                    {"content": r["content"], "metadata": r["metadata"], "similarity_score": r["similarity"]}
                    for r in results if r["similarity"] >= similarity_threshold
                ]
            
        except Exception as e:
            print(f"Error searching index for task {task_id}: {str(e)}")
            return []
    
    def search_by_cluster(self, query: str, task_id: str, cluster_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search within specific semantic cluster.
        
        Args:
            query: Search query
            task_id: Task identifier
            cluster_id: Specific cluster ID to search in
            
        Returns:
            List of results from the cluster
        """
        try:
            # Load index if not in memory
            if task_id not in self.indices:
                if not self.load_index(task_id):
                    return []
            
            metadata = self.index_metadata.get(task_id, {})
            clustering = metadata.get("clustering", {})
            clusters = clustering.get("clusters", {})
            
            if not clusters:
                return []
            
            # If no cluster specified, search all clusters
            if cluster_id is None:
                all_results = []
                for cid, cluster_chunks in clusters.items():
                    # Simple text matching within cluster
                    for chunk in cluster_chunks:
                        if query.lower() in chunk["content"].lower():
                            all_results.append({
                                "content": chunk["content"],
                                "metadata": chunk["metadata"],
                                "cluster_id": cid,
                                "match_type": "text"
                            })
                return all_results
            
            # Search specific cluster
            if cluster_id not in clusters:
                return []
            
            cluster_chunks = clusters[cluster_id]
            results = []
            
            for chunk in cluster_chunks:
                if query.lower() in chunk["content"].lower():
                    results.append({
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "cluster_id": cluster_id,
                        "match_type": "text"
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching cluster for task {task_id}: {str(e)}")
            return []
    
    def get_index_statistics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about the index.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Index statistics or None if not found
        """
        try:
            # Load index if not in memory
            if task_id not in self.indices:
                if not self.load_index(task_id):
                    return None
            
            metadata = self.index_metadata.get(task_id, {})
            clustering = metadata.get("clustering", {})
            
            return {
                "task_id": task_id,
                "total_chunks": metadata.get("total_chunks", 0),
                "embedding_model": metadata.get("embedding_model", "unknown"),
                "chunk_size": metadata.get("chunk_size", 0),
                "chunk_overlap": metadata.get("chunk_overlap", 0),
                "n_clusters": len(clustering.get("clusters", {})),
                "silhouette_score": clustering.get("silhouette_score", 0),
                "created_at": metadata.get("created_at", "unknown"),
                "index_type": metadata.get("index_type", "unknown")
            }
            
        except Exception as e:
            print(f"Error getting index statistics for task {task_id}: {str(e)}")
            return None
    
    def delete_index(self, task_id: str) -> bool:
        """
        Delete index files and clear from memory.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Remove from memory
            if task_id in self.indices:
                del self.indices[task_id]
            if task_id in self.index_metadata:
                del self.index_metadata[task_id]
            
            # Remove files
            index_file = os.path.join(self.vectorstore_dir, f"{task_id}.faiss")
            docs_map_file = os.path.join(self.vectorstore_dir, f"{task_id}_docs.json")
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            
            files_removed = 0
            for file_path in [index_file, docs_map_file, metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_removed += 1
            
            return files_removed > 0
            
        except Exception as e:
            print(f"Error deleting index for task {task_id}: {str(e)}")
            return False
