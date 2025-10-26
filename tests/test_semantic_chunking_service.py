import importlib
import pytest


class DummyEmbeddingService:
    def __init__(self):
        pass

    def embed_query(self, text: str):
        # return a deterministic small embedding based on text length
        return [float(len(text) % 7 + 1)] * 16


def test_semantic_chunking_creates_chunks(monkeypatch):
    sc_mod = importlib.import_module("app.services.semantic_chunking_service")
    # Monkeypatch the LocalEmbeddingService used inside the module
    monkeypatch.setattr(sc_mod, "LocalEmbeddingService", DummyEmbeddingService)

    svc = sc_mod.SemanticChunkingService()

    sample = (
        "Bonjour. Ceci est un test. "
        "Le modèle doit segmenter ce texte en phrases. "
        "Ensuite, il doit agréger des phrases similaires pour créer des chunks cohérents. "
        "Ce paragraphe parle de développement web et machine learning. "
        "La programmation Python est au coeur du ML. "
        "Enfin, on vérifie que le chunking retourne des chunks et des métadonnées."
    )

    chunks = svc.create_semantic_chunks(sample, task_id="t1")
    assert isinstance(chunks, list)
    # If chunks exist, each should have metadata with chunk_id and word_count
    if chunks:
        for c in chunks:
            assert "metadata" in c
            assert "chunk_id" in c["metadata"]
            assert "word_count" in c["metadata"]

    stats = svc.get_chunking_statistics(chunks)
    assert isinstance(stats, dict)
