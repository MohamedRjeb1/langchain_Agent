"""Combined RAG service: ingestion pipeline (YouTube -> transcript -> chunk -> embed -> index)
and retriever + query answering (corrective RAG + memory RAG simple implementation).

This service constructs its own retriever from the index managed by AdvancedIndexingService
and orchestrates LLM calls via LLMService.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from datetime import datetime

from app.core.config import get_settings
from app.services.indexing_service import AdvancedIndexingService
from app.services.embedding_service import LocalEmbeddingService
from app.services.llm_service import LLMService
from app.services.router_service import LLMBasedRouter
from app.services.semantic_chunking_service import SemanticChunkingService
from app.services.transcription_service import TranscriptionService
from app.services.youtube_service import YouTubeService
import numpy as np


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
        self.router = LLMBasedRouter(self.llm)

        mem_path = os.path.join(self.settings.DATA_DIR, "memory.json")
        self.memory = MemoryStore(persist_path=mem_path)

    # ---------- Chat Orchestration ----------
    def route_mode(self, latest_message: str, task_id: Optional[str], auto_retrieve: bool = True, force_rag: bool = False, chat_only: bool = False, llm_router: bool = False) -> Dict[str, Any]:
        """Simple router deciding between 'rag' and 'chat'. Heuristic-only by default.

        Returns: {mode: 'rag'|'chat', reason: str}
        """
        try:
            msg = (latest_message or "").lower()
            if chat_only:
                return {"mode": "chat", "reason": "chat_only toggle"}
            if force_rag:
                if task_id:
                    return {"mode": "rag", "reason": "force_rag toggle"}
                return {"mode": "chat", "reason": "force_rag but no task selected"}
            if not auto_retrieve:
                return {"mode": "chat", "reason": "auto_retrieve disabled"}
            if not task_id:
                return {"mode": "chat", "reason": "no task selected"}

            # LLM-based routing if requested and a task is selected
            if llm_router:
                try:
                    det = self.router.detect_mode(latest_message)
                    m = det.get("mode", "chat")
                    raw = det.get("raw", "")
                    if m == "rag":
                        return {"mode": "rag", "reason": f"llm_router: {raw}"}
                    return {"mode": "chat", "reason": f"llm_router: {raw}"}
                except Exception as e:
                    # Fall back to heuristics below
                    pass

            # Heuristic keywords
            keywords = [
                "video", "transcript", "dans la vidéo", "in the video", "speaker", "segment", "chunk",
                "ce que la vidéo dit", "what the video", "whisper", "youtube"
            ]
            if any(k in msg for k in keywords):
                return {"mode": "rag", "reason": "heuristic keywords matched"}

            # If question seems factual and specific, prefer RAG; else chat
            factual_signals = ["quel", "quelle", "combien", "when", "where", "who", "details", "exact"]
            if any(k in msg for k in factual_signals):
                return {"mode": "rag", "reason": "factual signal"}

            return {"mode": "chat", "reason": "default chat"}
        except Exception as e:
            return {"mode": "chat", "reason": f"router error: {e}"}

    def _build_chat_prompt(self, history: List[Dict[str, str]], latest_message: str, max_history: int = 8, do_summary: bool = True) -> str:
        """Construct a chat prompt from a short history window and optional summary."""
        hx = history[-max_history:]
        conv_lines = []
        for m in hx:
            role = m.get("role", "user")
            content = m.get("content", "")
            conv_lines.append(f"{role}: {content}")
        convo = "\n".join(conv_lines)
        prompt = (
            "You are a helpful and concise assistant. Answer naturally based on the ongoing conversation.\n"
            "Conversation so far:\n" + convo + "\n\n"
            "User: " + latest_message + "\nAssistant:"
        )
        # Minimal prompt; rolling summary can be added later if needed.
        return prompt

    def chat_respond(self, latest_message: str, history: List[Dict[str, str]], task_id: Optional[str] = None,
                     auto_retrieve: bool = True, force_rag: bool = False, chat_only: bool = False, llm_router: bool = False,
                     use_memory: bool = True, enable_corrective: bool = False,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """One chat turn: route to RAG or Chat, run, and return {mode_used, answer, provenance?}."""
        route = self.route_mode(latest_message, task_id, auto_retrieve=auto_retrieve, force_rag=force_rag, chat_only=chat_only, llm_router=llm_router)
        mode = route.get("mode", "chat")
        reason = route.get("reason", "")
        if progress_callback:
            progress_callback(f"[router] mode={mode} reason={reason}")

        if mode == "rag" and task_id:
            # delegate to existing RAG answer; do not inject history text, keep retrieval-only memory boost
            out = self.answer_query(task_id, latest_message, k=5, similarity_threshold=0.0, use_memory=use_memory, progress_callback=progress_callback, enable_corrective=enable_corrective)
            return {"mode_used": "rag", **out}
        else:
            # general chat
            prompt = self._build_chat_prompt(history, latest_message)
            if progress_callback:
                progress_callback("[llm] Generating chat response (no retrieval)")
            llm_out = self.llm.generate(prompt, max_tokens=400, temperature=0.2)
            return {"mode_used": "chat", "answer": llm_out.get("text", ""), "provenance": []}

    # ---------- Corrective RAG helpers ----------
    def _grade_knowledge_strips(self, query: str, strips: List[Dict[str, Any]], max_items: int = 8, progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Ask the LLM to grade each strip for relevance to the query.

        Returns a list of dicts: {chunk_id, grade (0-1), reason} aligned to input order when possible.
        Robust to non-JSON outputs by best-effort parsing.
        """
        try:
            items = []
            for s in strips[:max_items]:
                cid = (s.get("metadata") or {}).get("chunk_id", "")
                items.append({"chunk_id": cid, "text": s.get("content", "")})

            if not items:
                return []

            prompt = (
                "You are a strict relevance grader. For each knowledge strip, output a JSON array where each item has: "
                "{chunk_id: string, grade: number between 0 and 1, reason: short string}.\n"
                "Grade measures how well the strip helps answer the query.\n\n"
                f"Query: {query}\n\nKnowledge strips:\n"
            )
            for it in items:
                prompt += f"- [{it['chunk_id']}] {it['text'][:600]}\n"
            prompt += "\nReturn ONLY valid JSON (array)."

            if progress_callback:
                progress_callback("[llm] Grading retrieved knowledge strips for relevance")
            out = self.llm.generate(prompt, max_tokens=400, temperature=0.0)
            txt = out.get("text", "").strip()

            import json as _json
            import re as _re
            grades: List[Dict[str, Any]] = []
            try:
                # Be robust to code fences or extra text: extract the JSON array portion if present
                t = txt.strip()
                if t.startswith("```"):
                    # remove leading/trailing triple backticks blocks
                    t = t.strip('`')
                    # also try to drop a leading 'json' language tag
                    if t.lower().startswith("json\n"):
                        t = t[5:]
                m = _re.search(r"\[.*\]", t, _re.DOTALL)
                json_str = m.group(0) if m else t
                parsed = _json.loads(json_str)
                if isinstance(parsed, list):
                    for obj in parsed:
                        if not isinstance(obj, dict):
                            continue
                        cid = str(obj.get("chunk_id", ""))
                        try:
                            g = float(obj.get("grade", 0))
                        except Exception:
                            g = 0.0
                        reason = str(obj.get("reason", ""))
                        grades.append({"chunk_id": cid, "grade": max(0.0, min(1.0, g)), "reason": reason})
            except Exception:
                # Fallback: if JSON parsing fails, return empty grades to avoid blocking
                grades = []
                if progress_callback:
                    progress_callback("[llm] Could not parse grading JSON; defaulting grades to 0.0")

            # Align grades back to strips order; if missing, default low grade
            by_id = {g["chunk_id"]: g for g in grades}
            aligned: List[Dict[str, Any]] = []
            for s in strips[:max_items]:
                cid = (s.get("metadata") or {}).get("chunk_id", "")
                g = by_id.get(cid, {"grade": 0.0, "reason": "no-grade"})
                aligned.append({"chunk_id": cid, "grade": g.get("grade", 0.0), "reason": g.get("reason", "")})
            return aligned
        except Exception:
            return []

    def _rewrite_queries(self, query: str, n: int = 2, progress_callback: Optional[callable] = None) -> List[str]:
        """Ask the LLM to propose n concise reformulations of the query. Returns list of strings."""
        try:
            prompt = (
                f"Rewrite the following user query into {n} alternative concise searches that might retrieve better evidence.\n"
                "Return ONLY a JSON array of strings.\n\n"
                f"Query: {query}\n"
            )
            if progress_callback:
                progress_callback("[llm] Generating alternative query rewrites for corrective retrieval")
            out = self.llm.generate(prompt, max_tokens=200, temperature=0.2)
            txt = out.get("text", "").strip()
            import json as _json
            rewrites: List[str] = []
            try:
                parsed = _json.loads(txt)
                if isinstance(parsed, list):
                    rewrites = [str(x) for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                rewrites = []
            return rewrites[:n]
        except Exception:
            return []

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
            chunks = self.chunker.create_semantic_chunks(transcript, task_id=task_id, progress_callback=progress_callback)

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
            if progress_callback:
                progress_callback(f"[embed] Embeddings computed for {len(embeddings)} chunks")

            # Prepare embedding_results expected by AdvancedIndexingService
            embedding_results = []
            for doc, emb in zip(docs, embeddings):
                embedding_results.append({"document": doc, "embedding": emb})

            # Index
            if progress_callback:
                progress_callback("[index] Creating index")
            idx_res = self.indexer.create_hybrid_index(embedding_results, task_id)
            if progress_callback:
                chunks_n = idx_res.get("total_chunks") if isinstance(idx_res, dict) else None
                progress_callback(f"[index] Index created; total_chunks={chunks_n}")

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

    def answer_query(self, task_id: str, query: str, k: int = 5, similarity_threshold: float = 0.0, use_memory: bool = True, use_summary: bool = True, max_evidence_chars: int = 2000, progress_callback: Optional[callable] = None,
                     enable_corrective: bool = False, grade_threshold: float = 0.6, max_retrieval_rounds: int = 1, num_rewrites: int = 2,
                     memory_top_n: int = 2, memory_similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Answer a query using corrective RAG + optional memory.

        Returns: {answer, provenance: [chunks], used_memory: [...]}.
        """
        try:
            retriever = self.create_retriever(task_id)

            # ---------------- Memory-augmented retrieval (for precision only) ----------------
            # We will use past QA/queries to enhance retrieval, but we WON'T inject memory text into the prompt.
            base_results = retriever.retrieve(query, k=k, similarity_threshold=similarity_threshold)
            if progress_callback and base_results:
                progress_callback("[retrieval] Initial results:")
                for r in base_results[:10]:
                    cid = (r.get("metadata") or {}).get("chunk_id", "")
                    progress_callback(
                        f"  - id={cid} sim={r.get('similarity_score',0.0):.3f} "
                        f"faiss={r.get('faiss_similarity','-')} cl={r.get('cluster_id','-')}"
                    )

            enhanced_results: List[Dict[str, Any]] = list(base_results)
            mem_items = []
            similar_qa_summaries: List[str] = []  # no longer used in prompt, retained for compatibility
            mem_candidates: List[Tuple[float, Dict[str, Any], Optional[np.ndarray]]] = []  # (sim, mem, emb)
            if use_memory:
                # Fetch last 20 memory items
                mem_items = self.memory.get(task_id, limit=20)
                # Compute query embedding
                try:
                    q_emb_list = self.embedding.embed_query(query)
                    qv = np.asarray(q_emb_list, dtype=np.float32)
                    qv = qv / (np.linalg.norm(qv) + 1e-12)
                except Exception:
                    qv = None

                if qv is not None:
                    # Score memories by similarity to query
                    scored: List[Tuple[float, Dict[str, Any], Optional[np.ndarray]]] = []
                    for m in mem_items:
                        mtext = m.get("text") or ""
                        mmeta = m.get("metadata") or {}
                        mtype = mmeta.get("type")
                        if mtype not in ("qa", "query"):
                            continue
                        memb = m.get("embedding")
                        if memb is None:
                            try:
                                memb = self.embedding.embed_query(mtext)
                            except Exception:
                                memb = None
                        if memb is None:
                            continue
                        me = np.asarray(memb, dtype=np.float32)
                        me = me / (np.linalg.norm(me) + 1e-12)
                        sim = float(np.dot(qv, me))
                        sim = max(0.0, min(1.0, sim))
                        scored.append((sim, m, me))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    picked = [(s, m, e) for (s, m, e) in scored if s >= memory_similarity_threshold][:memory_top_n]
                    mem_candidates = picked

                    # Retrieval using a centroid embedding (query + memory)
                    if picked:
                        try:
                            mem_embs = [e for (_s, _m, e) in picked if e is not None]
                            if mem_embs:
                                mem_centroid = np.mean(np.stack(mem_embs, axis=0), axis=0)
                                alpha = 0.5  # memory influence
                                merged = qv + alpha * mem_centroid
                                merged = merged / (np.linalg.norm(merged) + 1e-12)
                                # use indexer directly to pass custom embedding
                                enhanced_via_centroid = self.indexer.search_similar(
                                    query="",
                                    task_id=task_id,
                                    k=k,
                                    similarity_threshold=similarity_threshold,
                                    query_embedding=merged.tolist(),
                                    embed_query=False,
                                )
                                enhanced_results.extend(enhanced_via_centroid)
                        except Exception:
                            pass

                    # Retrieval using memory-derived sub-queries (multi-query retrieval)
                    for (_s, m, _e) in mem_candidates:
                        # Try to extract the question part if present
                        mtext = m.get("text") or ""
                        qline = next((line.strip()[2:].strip() for line in mtext.splitlines() if line.strip().lower().startswith("q:")), None)
                        subq = qline if qline else (mtext[:200])
                        if not subq:
                            continue
                        try:
                            r2 = retriever.retrieve(subq, k=max(2, k//2), similarity_threshold=similarity_threshold)
                            enhanced_results.extend(r2)
                        except Exception:
                            continue

            # Deduplicate results by chunk_id, keeping the highest similarity score
            merged_map: Dict[str, Dict[str, Any]] = {}
            for r in enhanced_results:
                cid = (r.get("metadata") or {}).get("chunk_id", "")
                if not cid:
                    continue
                cur = merged_map.get(cid)
                if (cur is None) or (r.get("similarity_score", 0.0) > cur.get("similarity_score", 0.0)):
                    merged_map[cid] = r
            results = list(merged_map.values())
            results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

            # Initial graded list based on similarity (always available)
            graded = []
            for r in results:
                graded.append({
                    "content": r.get("content"),
                    "metadata": r.get("metadata"),
                    "score": r.get("similarity_score", 0.0),
                    "faiss_similarity": r.get("faiss_similarity"),
                    "cluster_id": r.get("cluster_id"),
                    "rank": r.get("rank"),
                })

            # Optional Corrective RAG: self-grade strips and retry retrieval if weak
            grade_info: List[Dict[str, Any]] = []
            if enable_corrective and graded:
                grade_info = self._grade_knowledge_strips(query, graded, max_items=min(8, len(graded)), progress_callback=progress_callback)
                # attach grade to graded entries
                by_id = {g["chunk_id"]: g for g in grade_info}
                for g in graded:
                    cid = (g.get("metadata") or {}).get("chunk_id", "")
                    gi = by_id.get(cid)
                    if gi:
                        g["grade"] = gi.get("grade", 0.0)
                        g["grade_reason"] = gi.get("reason", "")
                    else:
                        g["grade"] = 0.0
                        g["grade_reason"] = "no-grade"

                # Decide if we need corrective retrieval
                best_grade = max((g.get("grade", 0.0) for g in graded), default=0.0)
                if best_grade < grade_threshold:
                    # Try one or more retrieval rounds with rewrites
                    rounds = 0
                    while rounds < max_retrieval_rounds:
                        rounds += 1
                        rewrites = self._rewrite_queries(query, n=num_rewrites, progress_callback=progress_callback)
                        if progress_callback:
                            progress_callback(f"[retrieval] corrective round {rounds} rewrites={rewrites}")
                        merged: List[Dict[str, Any]] = list(graded)
                        seen_ids = {(m.get("metadata") or {}).get("chunk_id", "") for m in merged}
                        for q2 in rewrites:
                            if not q2:
                                continue
                            r2 = retriever.retrieve(q2, k=k, similarity_threshold=similarity_threshold)
                            if progress_callback and r2:
                                progress_callback(f"[retrieval] rewrite '{q2}' returned {len(r2)} results")
                            for rr in r2:
                                cid2 = (rr.get("metadata") or {}).get("chunk_id", "")
                                if cid2 in seen_ids:
                                    continue
                                merged.append({
                                    "content": rr.get("content"),
                                    "metadata": rr.get("metadata"),
                                    "score": rr.get("similarity_score", 0.0),
                                })
                                seen_ids.add(cid2)
                        # Re-grade on merged top items
                        grade_info = self._grade_knowledge_strips(query, merged, max_items=min(8, len(merged)), progress_callback=progress_callback)
                        by_id2 = {g["chunk_id"]: g for g in grade_info}
                        for g in merged:
                            cid = (g.get("metadata") or {}).get("chunk_id", "")
                            gi = by_id2.get(cid)
                            g["grade"] = gi.get("grade", 0.0) if gi else 0.0
                            g["grade_reason"] = gi.get("reason", "") if gi else "no-grade"
                        graded = merged
                        best_grade = max((g.get("grade", 0.0) for g in graded), default=0.0)
                        if best_grade >= grade_threshold:
                            break

                # After grading, prioritize by grade then similarity
                graded.sort(key=lambda x: (x.get("grade", 0.0), x.get("score", 0.0)), reverse=True)
                # Keep top-k
                graded = graded[:k]
                if progress_callback and graded:
                    progress_callback("[retrieval] Final selected strips (after grading/retrieval):")
                    for g in graded:
                        cid = (g.get("metadata") or {}).get("chunk_id", "")
                        progress_callback(
                            f"  - id={cid} rank={g.get('rank','-')} sim={g.get('score',0.0):.3f} "
                            f"grade={g.get('grade',0.0):.2f} cl={g.get('cluster_id','-')}"
                        )

            # We no longer append memory text to the prompt; memory is used only to enhance retrieval precision.

            # Build prompt: include top graded strips and memory
            prompt_parts = []
            # Do not include memory content in the prompt to comply with the new requirement.

            # Prepare evidence text and optionally summarize if too long
            # Build evidence text; if corrective grading present, annotate grade
            evidences = []
            for g in graded:
                cid = g['metadata'].get('chunk_id','')
                if enable_corrective and "grade" in g:
                    evidences.append(f"[{cid}] (grade {g.get('grade', 0.0):.2f}, sim {g.get('score', 0.0):.2f}, cl {g.get('cluster_id','-')}) {g['content']}")
                else:
                    evidences.append(f"[{cid}] (sim {g.get('score', 0.0):.2f}, cl {g.get('cluster_id','-')}) {g['content']}")
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
            try:
                # Store the query alone (lightweight)
                q_emb_store = None
                try:
                    q_emb_store = self.embedding.embed_query(query)
                except Exception:
                    q_emb_store = None
                self.memory.add(task_id, text=query, metadata={"type": "query"}, embedding=q_emb_store)
                # Store combined QA for future similarity
                qa_text = f"Q: {query}\nA: {answer_text or ''}"
                qa_emb = None
                try:
                    qa_emb = self.embedding.embed_query(qa_text)
                except Exception:
                    qa_emb = None
                self.memory.add(task_id, text=qa_text, metadata={"type": "qa"}, embedding=qa_emb)
            except Exception:
                # Non-fatal if persistence fails
                pass

            return {"answer": answer_text, "provenance": graded, "used_memory": mem_items}

        except Exception as e:
            return {"error": str(e)}
