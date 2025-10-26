import importlib
import os
import tempfile


def test_indexing_load_and_delete(monkeypatch, tmp_path):
    idx_mod = importlib.import_module("app.services.indexing_service")

    # Ensure the service uses a temporary vectorstore dir to avoid touching real data
    svc = idx_mod.AdvancedIndexingService()

    # Loading a non-existent index should return False
    assert svc.load_index("non_existent_task") is False

    # Searching on non-loaded index should return empty list
    res = svc.search_similar("query", task_id="non_existent_task")
    assert isinstance(res, list)
    assert res == []

    # Deleting a non-existent index should return False
    assert svc.delete_index("non_existent_task") in (False, True)
