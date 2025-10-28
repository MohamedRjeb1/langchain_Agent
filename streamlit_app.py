"""Streamlit prototype for the RAG system.

Usage: from project root run `streamlit run app/streamlit_app.py`.

This prototype is designed to work locally and uses the CombinedRAGService.
If heavy dependencies are missing, the app will show informative errors.
"""
import os
import sys
from pathlib import Path
import streamlit as st

from app.services.rag_service import CombinedRAGService


st.set_page_config(page_title="Local RAG Demo", layout="wide")

st.title("Local Video RAG — Prototype")

# Force a cache folder Streamlit/Whisper can read/write
WHISPER_CACHE = r"C:\Users\moham\OneDrive\Desktop\la\whisper_cache"
try:
    os.makedirs(WHISPER_CACHE, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE
except Exception as e:
    st.warning(f"[diagnostics] Could not ensure whisper cache directory: {e}")

with st.sidebar:
    st.header("Ingest video")
    youtube_url = st.text_input("YouTube URL", value="YouTube URL")
    task_id = st.text_input("Task ID", value="video_1")
    ingest_btn = st.button("Start ingestion")
    st.markdown("---")
    st.header("Query")
    use_memory = st.checkbox("Use memory", value=True)
    use_corrective = st.checkbox("Use Corrective RAG (self-grade & retry)", value=False)
    with st.expander("Diagnostics & Model Cache"):
        st.caption("Use this to validate the Python used by Streamlit and preload Whisper cache.")
        st.write("Python executable:")
        st.code(sys.executable, language="")
        st.write("Python version:")
        st.code(sys.version, language="")
        st.write("Whisper cache directory:")
        st.code(WHISPER_CACHE, language="")
        preload = st.button("Preload Whisper 'small'")
        if preload:
            with st.spinner("Loading Whisper model 'small' (may download on first use)..."):
                try:
                    # import whisper lazily inside the action
                    import whisper
                    model = whisper.load_model("small")
                    st.success("Whisper model loaded successfully and cached.")
                except Exception as e:
                    st.error(f"Failed to preload Whisper: {e}")


@st.cache_resource
def get_service():
    # instantiate lazily but cached across reruns
    return CombinedRAGService()


# We'll create the service lazily when a button is pressed to avoid model downloads on page load.
svc = None

# Simple in-page log area
logs: list[str] = []
log_box = st.empty()


def append_log(msg: str):
    from datetime import datetime

    ts = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{ts}] {msg}")
    # keep only last 200 lines
    display = "\n".join(logs[-200:])
    # show logs in a monospace block
    log_box.code(display, language="")


st.sidebar.markdown("\n---\nLogs (shows download/model/load steps and progress). If you see large downloads, check the log entries to determine if it's a model or media file.")


if ingest_btn:
    # instantiate service here to avoid eager model loads
    try:
        svc = get_service()
    except Exception as e:
        append_log(f"[error] Failed to initialize RAG service: {e}")
        st.error("Failed to initialize RAG service; check logs in the sidebar.")

    if svc:
        append_log(f"[ui] Starting ingestion for {youtube_url} -> task {task_id}")
        with st.spinner("Ingesting video — check logs for detailed progress..."):
            res = svc.ingest_from_youtube(youtube_url, task_id=task_id, progress_callback=append_log)

        if res.get("status") == "completed":
            append_log("[ui] Ingestion completed successfully")
            st.success("Ingestion completed")
            st.json(res.get("index_result"))
        else:
            append_log(f"[ui] Ingestion failed: {res}")
            st.error(f"Ingestion failed: {res}")

st.markdown("---")

st.header("Chat with the video")
query = st.text_input("Ask a question about the video")
ask_btn = st.button("Ask")
if ask_btn and query:
    # ensure service exists
    if svc is None:
        try:
            svc = get_service()
        except Exception as e:
            append_log(f"[error] Failed to initialize RAG service: {e}")
            st.error("Failed to initialize RAG service; check logs in the sidebar.")

    if svc is not None:
        append_log(f"[ui] Running query for task {task_id}: {query}")
        with st.spinner("Retrieving and answering — check logs for model/load progress..."):
            out = svc.answer_query(task_id, query, k=5, similarity_threshold=0.0, use_memory=use_memory, progress_callback=append_log, enable_corrective=use_corrective)

        if out.get("error"):
            append_log(f"[ui] Query error: {out.get('error')}")
            st.error(out.get("error"))
        else:
            append_log("[ui] Query answered")
            st.markdown("### Answer")
            st.write(out.get("answer"))
            st.markdown("### Provenance")
            for p in out.get("provenance", []):
                cid = p.get('metadata',{}).get('chunk_id','')
                grade = p.get('grade', None)
                sim = p.get('score', None)
                prefix = f"- {cid}"
                if grade is not None or sim is not None:
                    gtxt = f" grade={grade:.2f}" if isinstance(grade, (int, float)) else ""
                    stxt = f" sim={sim:.2f}" if isinstance(sim, (int, float)) else ""
                    prefix += f" ({gtxt}{',' if gtxt and stxt else ''}{stxt})"
                st.write(f"{prefix} : {p.get('content')[:300]}")
            st.markdown("### Used memory")
            for m in out.get("used_memory", []):
                st.write(f"- {m.get('created_at')}: {m.get('text')}")
