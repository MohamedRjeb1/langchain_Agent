import importlib


def test_combined_rag_ingest_and_query(monkeypatch):
    rag_mod = importlib.import_module("app.services.rag_service")

    # Monkeypatch YouTubeService.download_video
    class DummyYT:
        def download_video(self, url, task_id):
            return {"status": "completed", "audio_file": "fake.mp3"}

    monkeypatch.setattr(rag_mod, "YouTubeService", lambda: DummyYT())

    # Monkeypatch TranscriptionService
    class DummyTS:
        def transcribe_audio(self, audio_file, task_id, language=None):
            return {"status": "completed", "transcript": "This is a test transcript."}

    monkeypatch.setattr(rag_mod, "TranscriptionService", lambda: DummyTS())

    # Monkeypatch SemanticChunkingService
    class DummyChunker:
        def create_semantic_chunks(self, text, task_id=None):
            return [{"content": "chunk1 text", "metadata": {"chunk_id": "c1"}},
                    {"content": "chunk2 text", "metadata": {"chunk_id": "c2"}}]

    monkeypatch.setattr(rag_mod, "SemanticChunkingService", lambda: DummyChunker())

    # Monkeypatch EmbeddingService
    class DummyEmbed:
        def embed_documents(self, texts):
            return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(rag_mod, "LocalEmbeddingService", lambda: DummyEmbed())

    # Monkeypatch AdvancedIndexingService
    class DummyIndexer:
        def __init__(self):
            self.indices = {}

        def create_hybrid_index(self, embedding_results, task_id):
            # pretend to create index
            self.indices[task_id] = embedding_results
            return {"status": "completed", "task_id": task_id}

        def load_index(self, task_id):
            return task_id in self.indices

        def search_similar(self, query, task_id, k=5, similarity_threshold=0.0):
            # return top chunk as found
            return [{"content": "chunk1 text", "metadata": {"chunk_id": "c1"}, "similarity_score": 0.95}]

    monkeypatch.setattr(rag_mod, "AdvancedIndexingService", lambda: DummyIndexer())

    # Monkeypatch LLMService
    class DummyLLM:
        def __init__(self, model_name="mistral"):
            pass

        def generate(self, prompt, max_tokens=512, temperature=0.0):
            return {"text": "This is a dummy answer."}

    monkeypatch.setattr(rag_mod, "LLMService", lambda model_name=None: DummyLLM())

    # Now instantiate CombinedRAGService and run ingestion + query
    svc = rag_mod.CombinedRAGService()

    ingest_res = svc.ingest_from_youtube("https://youtube.test/video", task_id="t1")
    assert ingest_res.get("status") == "completed"

    ans = svc.answer_query("t1", "What is this video about?", k=2)
    assert "answer" in ans
    assert ans["answer"] == "This is a dummy answer."
