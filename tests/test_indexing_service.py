"""Tests for AdvancedIndexingService (FAISS backend)."""





if __name__ == "__main__":
    # Manual run placeholder
    pass


# Real pytest-based test for the FAISS-backed indexer
def test_indexing_service_create_search_and_persist():
    import os
    import shutil
    import tempfile
    from app.services.indexing_service import AdvancedIndexingService

    class Doc:
        def __init__(self, text, metadata=None):
            self.page_content = text
            self.metadata = metadata or {}

    tmpdir = tempfile.mkdtemp(prefix="indexer_test_")
    try:
        svc = AdvancedIndexingService()
        # redirect vectorstore to tmp to avoid polluting workspace
        svc.vectorstore_dir = tmpdir
        os.makedirs(svc.vectorstore_dir, exist_ok=True)

        # Build simple dataset with 3 docs in 4D
        embs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        docs = [
            Doc("alpha", {"chunk_id": 0}),
            Doc("beta", {"chunk_id": 1}),
            Doc("gamma", {"chunk_id": 2}),
        ]
        embedding_results = [{"document": d, "embedding": e} for d, e in zip(docs, embs)]

        task_id = "t_index"
        out = svc.create_hybrid_index(embedding_results, task_id)
        assert out["status"] == "completed"
        assert os.path.exists(os.path.join(tmpdir, f"{task_id}.faiss"))

        # Query using direct embedding to avoid external embedder
        q = [1.0, 0.0, 0.0, 0.0]
        res = svc.search_similar(query="alpha", task_id=task_id, k=2, similarity_threshold=0.5, query_embedding=q, embed_query=False)
        assert len(res) >= 1
        assert res[0]["metadata"]["chunk_id"] == 0
        assert 0.9 <= res[0]["similarity_score"] <= 1.0

        # Load in new instance and search again
        svc2 = AdvancedIndexingService()
        svc2.vectorstore_dir = tmpdir
        assert svc2.load_index(task_id) is True
        res2 = svc2.search_similar(query="alpha", task_id=task_id, k=2, similarity_threshold=0.5, query_embedding=q, embed_query=False)
        assert len(res2) >= 1
        assert res2[0]["metadata"]["chunk_id"] == 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)