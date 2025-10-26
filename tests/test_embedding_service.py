import builtins
import types
import pytest


def test_local_embedding_service_monkeypatched(monkeypatch):
    # Create a dummy OllamaEmbeddings replacement
    class DummyOllama:
        def __init__(self, model=None):
            self.model = model

        def embed_query(self, text: str):
            # deterministic small vector based on text length
            n = 8
            return [float(len(text) % 10 + 1)] * n

        def embed_documents(self, texts):
            return [[float(len(t) % 10 + 1)] * 8 for t in texts]

    # Monkeypatch the imported class in the service module
    import importlib
    es_mod = importlib.import_module("app.services.embedding_service")
    monkeypatch.setattr(es_mod, "OllamaEmbeddings", DummyOllama)

    LocalEmbeddingService = es_mod.LocalEmbeddingService
    svc = LocalEmbeddingService()

    v = svc.embed_query("hello world")
    assert isinstance(v, list)
    assert len(v) == svc.get_embedding_dimension() or len(v) == 8

    docs = ["a", "bb", "ccc"]
    embs = svc.embed_documents(docs)
    assert isinstance(embs, list)
    assert len(embs) == len(docs)

    # cache behaviour: repeated query should return same vector value
    v2 = svc.embed_query("hello world")
    assert v == v2
