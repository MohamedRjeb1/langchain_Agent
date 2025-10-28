import os
import json
import shutil
import tempfile
import numpy as np

from app.services.faiss_index import FAISSIndex, l2_normalize


def test_faiss_index_add_search_save_load():
    tmp = tempfile.mkdtemp(prefix="faiss_test_")
    try:
        dim = 4
        idx = FAISSIndex(dim=dim, index_dir=tmp, task_id="t1")
        embs = np.eye(dim, dtype=np.float32)  # 4 vectors e1..e4
        docs = [{"content": f"doc{i}", "metadata": {"i": i}} for i in range(dim)]

        added_ids = idx.add(embs, docs)
        assert len(added_ids) == dim
        assert idx.size == dim

        # Query close to e0
        q = [1.0, 0.0, 0.0, 0.0]
        res = idx.search(q, k=3)
        assert len(res) >= 1
        # Top-1 should be doc0 with high sim
        assert res[0]["content"] == "doc0"
        assert 0.99 <= res[0]["similarity"] <= 1.0

        # Save and reload
        idx.save()
        idx2 = FAISSIndex(dim=dim, index_dir=tmp, task_id="t1")
        assert idx2.load() is True
        res2 = idx2.search(q, k=3)
        assert res2[0]["content"] == "doc0"
        assert 0.99 <= res2[0]["similarity"] <= 1.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
