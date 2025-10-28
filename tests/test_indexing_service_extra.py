import os
import shutil
import tempfile
import threading
import time

import numpy as np

from app.services.faiss_index import FAISSIndex
from app.services.indexing_service import AdvancedIndexingService


def test_concurrent_saves_are_safe():
    tmpdir = tempfile.mkdtemp(prefix="faiss_concurrent_")
    try:
        dim = 8
        idx = FAISSIndex(dim=dim, index_dir=tmpdir, task_id="t_conc")
        embs = np.eye(dim, dtype=np.float32)
        docs = [{"content": f"d{i}", "metadata": {"i": i}} for i in range(dim)]
        idx.add(embs, docs)

        def worker_save(n=3):
            for _ in range(n):
                idx.save()

        threads = [threading.Thread(target=worker_save) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should be able to load cleanly
        idx2 = FAISSIndex(dim=dim, index_dir=tmpdir, task_id="t_conc")
        assert idx2.load() is True
        assert idx2.size == idx.size
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_query_embedding_consistency(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="indexer_embed_parity_")
    try:
        svc = AdvancedIndexingService()
        svc.vectorstore_dir = tmpdir
        os.makedirs(svc.vectorstore_dir, exist_ok=True)

        # Dataset with simple 4D one-hot embeddings
        class Doc:
            def __init__(self, txt, meta=None):
                self.page_content = txt
                self.metadata = meta or {}

        docs = [
            Doc("alpha", {"chunk_id": 0}),
            Doc("beta", {"chunk_id": 1}),
        ]
        embs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        embedding_results = [{"document": d, "embedding": e} for d, e in zip(docs, embs)]
        svc.create_hybrid_index(embedding_results, task_id="t_parity")

        # Monkeypatch embedder to return the same vector as we pass explicitly
        class DummyEmbedder:
            def embed_query(self, text: str):
                return [1.0, 0.0, 0.0, 0.0]

        svc.embedder = DummyEmbedder()

        # A: with embedder
        A = svc.search_similar(query="alpha", task_id="t_parity", k=1, similarity_threshold=0.0, query_embedding=None, embed_query=True)
        # B: bypass embedder with explicit vector
        B = svc.search_similar(query="alpha", task_id="t_parity", k=1, similarity_threshold=0.0, query_embedding=[1.0, 0.0, 0.0, 0.0], embed_query=False)

        assert A and B
        assert A[0]["metadata"]["chunk_id"] == B[0]["metadata"]["chunk_id"] == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
