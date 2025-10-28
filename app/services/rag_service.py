"""Combined RAG service: ingestion pipeline (YouTube -> transcript -> chunk -> embed -> index)
and retriever + query answering (corrective RAG + memory RAG simple implementation).

This service constructs its own retriever from the index managed by AdvancedIndexingService
and orchestrates LLM calls via LLMService.
"""
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

from app.core.config import get_settings
from app.services.indexing_service import AdvancedIndexingService
from app.services.embedding_service import LocalEmbeddingService
from app.services.llm_service import LLMService
from app.services.semantic_chunking_service import SemanticChunkingService
from app.services.transcription_service import TranscriptionService
from app.services.youtube_service import YouTubeService


class SimpleRetriever:
    """A thin retriever wrapper that uses an AdvancedIndexingService instance's search_similar.

    This keeps retriever creation inside the RAG service as requested by the user.
    """

    def __init__(self, indexer: AdvancedIndexingService, task_id: str):
        self.indexer = indexer
        self.task_id = task_id

    def retrieve(self, query: str, k: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        return self.indexer.search_similar(query, task_id=self.task_id, k=k, similarity_threshold=similarity_threshold)


class MemoryStore:
    """Simple in-memory (and optionally on-disk) memory store keyed by task_id/user.

    Stores list of memories: {text, metadata, created_at, embedding}
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.memory: Dict[str, List[Dict[str, Any]]] = {}
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            try:
                with open(persist_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = {}

    def add(self, task_id: str, text: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None):
        entry = {
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
        }
        self.memory.setdefault(task_id, []).append(entry)
        if self.persist_path:
            try:
                with open(self.persist_path, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def get(self, task_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.memory.get(task_id, []))[-limit:]


class CombinedRAGService:
    def __init__(self):
        self.settings = get_settings()
        self.youtube = YouTubeService()
        self.transcriber = TranscriptionService()
        self.chunker = SemanticChunkingService()
        self.embedding = LocalEmbeddingService()
        self.indexer = AdvancedIndexingService()
        self.llm = LLMService(model_name=self.settings.DEFAULT_LLM_MODEL)

        mem_path = os.path.join(self.settings.DATA_DIR, "memory.json")
        self.memory = MemoryStore(persist_path=mem_path)

    def ingest_from_youtube(self, url: str, task_id: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Full ingestion flow for a YouTube URL.

        Steps: download audio -> transcribe -> chunk -> embed -> index
        Returns a dict with status and useful paths/metadata.
        """
        try:
            if progress_callback:
                progress_callback(f"[ingest] Starting ingestion for task {task_id}")

            # prepare a yt-dlp progress hook that forwards events to progress_callback
            def _yt_progress_hook(d: dict):
                # d can contain keys: status, filename, downloaded_bytes, total_bytes, eta
                status = d.get("status")
                if status == "downloading":
                    downloaded = d.get("downloaded_bytes") or d.get("downloaded_bytes", 0)
                    total = d.get("total_bytes") or d.get("total_bytes", 0)
                    percent = None
                    try:
                        if total:
                            percent = downloaded / total * 100
                    except Exception:
                        percent = None
                    msg = f"[download] {d.get('filename','')} - {downloaded}/{total} bytes"
                    if percent is not None:
                        msg = f"[download] {percent:.1f}% - {downloaded}/{total} bytes"
                    if progress_callback:
                        progress_callback(msg)
                elif status == "finished":
                    if progress_callback:
                        progress_callback(f"[download] Finished: {d.get('filename','')}")

            # try to use progress_hook if the youtube service supports it; fall back otherwise
            try:
                dl = self.youtube.download_video(url, task_id, progress_hook=_yt_progress_hook)
            except TypeError:
                # older signature without progress_hook
                if progress_callback:
                    progress_callback("[download] progress hook not supported by youtube service, starting download without progress updates")
                dl = self.youtube.download_video(url, task_id)
            if dl.get("status") != "completed":
                # If the youtube service returns FAILED, propagate
                return {"status": dl.get("status"), "message": dl.get("message"), "error": dl.get("error")}

            audio_file = dl.get("audio_file")
            # Transcribe
            if progress_callback:
                progress_callback("[transcribe] Starting transcription (Whisper model may load now)")
            # Be compatible with older signature without progress_callback
            try:
                tr = self.transcriber.transcribe_audio(audio_file, task_id, progress_callback=progress_callback)
            except TypeError:
                tr = self.transcriber.transcribe_audio(audio_file, task_id)
            if tr.get("status") != "completed":
                return {"status": tr.get("status"), "message": tr.get("message"), "error": tr.get("error")}

            transcript = tr.get("transcript")

            # Chunk
            if progress_callback:
                progress_callback("[chunk] Creating semantic chunks")
            chunks = self.chunker.create_semantic_chunks(transcript, task_id=task_id)

            # Create Document objects expected by indexer: reuse existing chunk dicts
            docs = []
            texts = []
            for c in chunks:
                # Create minimal document-like object with page_content and metadata
                doc = type("Doc", (), {})()
                doc.page_content = c["content"]
                doc.metadata = c["metadata"]
                docs.append(doc)
                texts.append(c["content"])

            # Embed
            if progress_callback:
                progress_callback("[embed] Embedding documents (embedding model may load now)")
            embeddings = self.embedding.embed_documents(texts)

            # Prepare embedding_results expected by AdvancedIndexingService
            embedding_results = []
            for doc, emb in zip(docs, embeddings):
                embedding_results.append({"document": doc, "embedding": emb})

            # Index
            if progress_callback:
                progress_callback("[index] Creating index")
            idx_res = self.indexer.create_hybrid_index(embedding_results, task_id)

            # Optionally add a summary memory entry
            summary_text = "".join([t[:400] + "..." for t in texts[:3]])
            self.memory.add(task_id, summary_text, metadata={"type": "ingest_summary"}, embedding=None)

            if progress_callback:
                progress_callback(f"[ingest] Ingestion completed for task {task_id}")

            return {"status": "completed", "index_result": idx_res}

        except Exception as e:
            return {"status": "failed", "message": str(e)}

    def create_retriever(self, task_id: str):
        """Return a SimpleRetriever bound to the indexer's task_id.

        The retriever provides a `retrieve(query, k, similarity_threshold)` method.
        """
        # Load index if necessary
        if task_id not in self.indexer.indices:
            loaded = self.indexer.load_index(task_id)
            if not loaded:
                raise RuntimeError(f"Index for task {task_id} not found")

        return SimpleRetriever(self.indexer, task_id)

    def answer_query(self, task_id: str, query: str, k: int = 5, similarity_threshold: float = 0.0, use_memory: bool = True, use_summary: bool = True, max_evidence_chars: int = 2000, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Answer a query using corrective RAG + optional memory.

        Returns: {answer, provenance: [chunks], used_memory: [...]}.
        """
        try:
            retriever = self.create_retriever(task_id)
            results = retriever.retrieve(query, k=k, similarity_threshold=similarity_threshold)

            # Grade strips: results already filtered by threshold in indexer.search_similar
            graded = []
            for r in results:
                graded.append({
                    "content": r.get("content"),
                    "metadata": r.get("metadata"),
                    "score": r.get("similarity_score", 0.0),
                })

            # If no high-quality strips, optionally include memory or return a message
            mem_items = []
            if use_memory:
                mem_items = self.memory.get(task_id, limit=5)

            # Build prompt: include top graded strips and memory
            prompt_parts = []
            if mem_items:
                prompt_parts.append("Memory context:\n")
                for m in mem_items:
                    prompt_parts.append(f"- {m.get('text')}\n")

            # Prepare evidence text and optionally summarize if too long
            evidences = [f"[{g['metadata'].get('chunk_id','')}] {g['content']}" for g in graded]
            evidence_text = "\n".join(evidences)
            if use_summary and len(evidence_text) > max_evidence_chars:
                if progress_callback:
                    progress_callback("[llm] Summarizing retrieved evidence to fit token budget")
                summary_prompt = (
                    "Summarize the following retrieved evidence into a concise digest that preserves key facts and names.\n"
                    "Use bullet points where appropriate, keep it under 500 words.\n\n" + evidence_text
                )
                try:
                    sum_out = self.llm.generate(summary_prompt, max_tokens=400, temperature=0.0)
                    evidence_text = sum_out.get("text") or evidence_text[:max_evidence_chars]
                except Exception:
                    # Fallback to truncation if LLM summarization fails
                    evidence_text = evidence_text[:max_evidence_chars]

            prompt_parts.append("Retrieved evidence:\n")
            prompt_parts.append(evidence_text + "\n")

            prompt_parts.append("User question:\n" + query + "\n")
            prompt = "\n".join(prompt_parts)

            # Call LLM
            if progress_callback:
                progress_callback("[llm] Calling LLM to generate answer (model may be downloaded/loaded now)")
            llm_out = self.llm.generate(prompt, max_tokens=512, temperature=0.0)
            answer_text = llm_out.get("text")

            # Store Q/A to memory
            self.memory.add(task_id, text=query, metadata={"type": "query"}, embedding=None)

            return {"answer": answer_text, "provenance": graded, "used_memory": mem_items}

        except Exception as e:
            return {"error": str(e)}
