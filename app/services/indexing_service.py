"""
Advanced indexing service with innovative techniques.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle
from collections import defaultdict

try:
    from langchain.vectorstores import DocArrayInMemorySearch
except Exception:  # pragma: no cover - optional dependency may be missing in test env
    DocArrayInMemorySearch = None

try:
    from langchain.schema import Document
except Exception:  # pragma: no cover - optional dependency may be missing in test env
    Document = None

# Prefer langchain_core vectorstore if available (newer lightweight API)
try:
    from langchain_core.vectorstores import InMemoryVectorStore
except Exception:  # pragma: no cover
    InMemoryVectorStore = None

try:
    # DocArrayRetriever may live in langchain_core.retrievers or langchain_core.vectorstores depending on version
    from langchain_core.retrievers import DocArrayRetriever
except Exception:
    try:
        from langchain_core.vectorstores import DocArrayRetriever
    except Exception:
        DocArrayRetriever = None


    # Simple fallback vectorstore for environments without langchain/docarray
    class SimpleInMemoryVectorStore:
        """A minimal in-memory vector store used as a fallback for tests/local runs.

        It stores documents and their embeddings and supports a basic
        similarity_search_with_score(query, k) method. If the query is a string,
        similarity is computed via simple token overlap; if the query is a vector,
        cosine similarity with stored embeddings is used.
        """

        def __init__(self, documents: List[Any], embeddings: List[List[float]]):
            # documents: list of objects with .page_content and .metadata or dicts
            self.documents = documents
            self.embeddings = np.array(embeddings, dtype=float) if embeddings else np.array([])

        def _text_similarity(self, q: str, text: str) -> float:
            qset = set(q.lower().split())
            tset = set(text.lower().split())
            if not qset or not tset:
                return 0.0
            inter = qset.intersection(tset)
            return len(inter) / max(1, min(len(qset), len(tset)))

        def similarity_search_with_score(self, query, k: int = 5):
            # If query is vector-like
            try:
                q = np.array(query, dtype=float)
                if self.embeddings.size == 0:
                    return []
                # cosine similarity
                dots = (self.embeddings @ q).astype(float)
                norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q) + 1e-12)
                sims = dots / (norms + 1e-12)
                # convert to distances (1 - sim) to match expected interface
                idx = np.argsort(-sims)[:k]
                results = []
                for i in idx:
                    doc = self.documents[i]
                    score = 1 - float(sims[i])
                    results.append((doc, score))
                return results
            except Exception:
                # treat query as text
                qtext = str(query)
                scores = []
                for i, doc in enumerate(self.documents):
                    content = getattr(doc, 'page_content', None) or (doc.get('page_content') if isinstance(doc, dict) else str(doc))
                    sim = self._text_similarity(qtext, content)
                    # convert to distance-like score: smaller is better
                    score = 1 - sim
                    scores.append((i, score))
                scores.sort(key=lambda x: x[1])
                results = []
                for i, score in scores[:k]:
                    results.append((self.documents[i], float(score)))
                return results


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
        
        # Index storage
        self.indices = {}
        self.index_metadata = {}
    
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
            
            # Organize results by cluster
            clusters = defaultdict(list)
            for i, (result, label) in enumerate(zip(embedding_results, cluster_labels)):
                clusters[label].append({
                    "chunk_id": result["document"].metadata["chunk_id"],
                    "content": result["document"].page_content,
                    "metadata": result["document"].metadata,
                    "embedding": result["embedding"]
                })
            
            return {
                "clusters": dict(clusters),
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
            # Create documents for vector store
            documents = []
            embeddings = []
            for result in embedding_results:
                doc = result["document"]
                documents.append(doc)
                embeddings.append(result.get("embedding"))

            # Create vector store using langchain_core InMemoryVectorStore if available,
            # otherwise prefer DocArrayInMemorySearch, else fallback to simple in-memory store
            if InMemoryVectorStore is not None:
                try:
                    # InMemoryVectorStore often expects a list/array of embeddings or (docs, embeddings)
                    try:
                        # try constructor signature InMemoryVectorStore(embeddings)
                        vectorstore = InMemoryVectorStore(embeddings)
                    except Exception:
                        # try (documents, embeddings) signature
                        vectorstore = InMemoryVectorStore(documents, embeddings)
                except Exception:
                    # fallback
                    vectorstore = SimpleInMemoryVectorStore(documents, embeddings)
            elif DocArrayInMemorySearch is not None:
                try:
                    vectorstore = DocArrayInMemorySearch.from_documents(
                        documents,
                        embedding=None,  # We'll use our custom embeddings
                    )
                except Exception:
                    vectorstore = SimpleInMemoryVectorStore(documents, embeddings)
            else:
                vectorstore = SimpleInMemoryVectorStore(documents, embeddings)
            
            # Create semantic clusters
            clustering_result = self.create_semantic_clusters(embedding_results)
            
            # Create index metadata
            index_metadata = {
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "total_chunks": len(embedding_results),
                "embedding_model": self.settings.DEFAULT_EMBEDDING_MODEL,
                "chunk_size": self.settings.CHUNK_SIZE,
                "chunk_overlap": self.settings.CHUNK_OVERLAP,
                "clustering": clustering_result,
                "index_type": "hybrid"
            }
            
            # Save index
            index_file = os.path.join(self.vectorstore_dir, f"{task_id}_index.pkl")
            with open(index_file, "wb") as f:
                pickle.dump(vectorstore, f)
            
            # Save metadata
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(index_metadata, f, indent=2, ensure_ascii=False)
            
            # Store in memory
            self.indices[task_id] = vectorstore
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
            index_file = os.path.join(self.vectorstore_dir, f"{task_id}_index.pkl")
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                return False
            
            # Load index
            with open(index_file, "rb") as f:
                vectorstore = pickle.load(f)
            
            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Store in memory
            self.indices[task_id] = vectorstore
            self.index_metadata[task_id] = metadata
            
            return True
            
        except Exception as e:
            print(f"Error loading index for task {task_id}: {str(e)}")
            return False
    
    def search_similar(self, query: str, task_id: str, k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using hybrid approach.
        
        Args:
            query: Search query
            task_id: Task identifier
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Load index if not in memory
            if task_id not in self.indices:
                if not self.load_index(task_id):
                    return []
            
            vectorstore = self.indices[task_id]

            # Perform similarity search — try common method names used by possible vectorstores
            results = []
            if hasattr(vectorstore, "similarity_search_with_score"):
                results = vectorstore.similarity_search_with_score(query, k=k)
            elif hasattr(vectorstore, "similarity_search"):
                # some stores return docs only
                docs = vectorstore.similarity_search(query, k=k)
                results = [(d, 0.0) for d in docs]
            elif hasattr(vectorstore, "search"):
                docs = vectorstore.search(query, k=k)
                results = [(d, 0.0) for d in docs]
            elif hasattr(vectorstore, "invoke"):
                # langchain_core retriever style
                try:
                    doc = vectorstore.invoke(query)
                    results = [(doc, 0.0)] if doc else []
                except Exception:
                    results = []
            else:
                results = []
            
            # Filter by similarity threshold and format results
            filtered_results = []
            for doc, score in results:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                
                if similarity >= similarity_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": similarity,
                        "distance_score": score
                    })
            
            return filtered_results
            
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

    def create_retriever(self, task_id: str, embeddings: Optional[Any] = None, **kwargs):
        """Create a retriever for a given index/task.

        If `DocArrayRetriever` from langchain_core is available, build and return it.
        Otherwise, return the in-memory vectorstore instance or None.
        """
        if task_id not in self.indices:
            if not self.load_index(task_id):
                return None

        vectorstore = self.indices[task_id]

        # If langchain_core's DocArrayRetriever is available and the vectorstore matches
        if DocArrayRetriever is not None:
            try:
                # Attempt to create a retriever using the vectorstore and provided embeddings
                retriever = DocArrayRetriever(index=vectorstore, embeddings=embeddings, **kwargs)
                return retriever
            except Exception:
                pass

        # Fallback: if vectorstore has an `invoke` or `similarity_search_with_score`, return it as-is
        return vectorstore

    
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
            index_file = os.path.join(self.vectorstore_dir, f"{task_id}_index.pkl")
            metadata_file = os.path.join(self.vectorstore_dir, f"{task_id}_index_metadata.json")
            
            files_removed = 0
            for file_path in [index_file, metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_removed += 1
            
            return files_removed > 0
            
        except Exception as e:
            print(f"Error deleting index for task {task_id}: {str(e)}")
            return False
