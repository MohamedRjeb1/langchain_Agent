"""Streamlit chat UI for local RAG over YouTube videos.

This interface provides a modern chat experience. Every user message is answered via
grounded retrieval (RAG). Memory and Corrective RAG are always enabled to maximize
retrieval precision and answer quality. A live log panel on the right shows progress.
"""
import os
import sys
import streamlit as st

from app.services.rag_service import CombinedRAGService


st.set_page_config(page_title="vidéo into Advanced RAG", layout="wide")

# --- Global CSS: gradient hero, glass panels, buttons, focus, responsive ---
st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root { --bg:#0b1220; --panel:rgba(15,23,42,0.6); --text:#e5e7eb; --muted:#94a3b8; --card:rgba(255,255,255,0.08); --border:rgba(148,163,184,0.25); --accent1:#6366F1; --accent2:#06B6D4; --radius-2xl:26px; --shadow:0 10px 25px rgba(0,0,0,0.18); }
            html, body, [data-testid="stAppViewContainer"] { background: radial-gradient(1200px 800px at 20% -10%, rgba(99,102,241,0.25), transparent 60%), radial-gradient(1000px 600px at 120% 20%, rgba(6,182,212,0.22), transparent 60%), linear-gradient(180deg, #0b1220 0%, #0b1220 100%); color: var(--text); font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
            .block-container { padding-top: 1rem; }
            [data-testid="stSidebar"] { width:300px; min-width:300px; }
            .card { border-radius: var(--radius-2xl); background: var(--card); border: 1px solid var(--border); box-shadow: var(--shadow); padding: 16px 18px; backdrop-filter: blur(8px); }
            .sidebar-info { font-size: 0.9rem; color: #10b981; }
            .logs-box { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12.5px; white-space: pre-wrap; max-height: 520px; overflow: auto; padding: 10px 12px; border-radius: 16px; background: rgba(2,6,23,0.6); border: 1px solid var(--border); }
            .hero { text-align:center; padding: 32px 16px 18px; border-radius: var(--radius-2xl); background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(6,182,212,0.12)); border: 1px solid var(--border); box-shadow: var(--shadow); backdrop-filter: blur(10px); }
            .hero-title { font-size: clamp(32px, 6vw, 56px); font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(90deg, #fff, #dbeafe 30%, #93c5fd 60%, #a5b4fc 90%); -webkit-background-clip:text; background-clip:text; color:transparent; }
            .hero-underline { width: 220px; height: 4px; margin: 12px auto 0; border-radius:999px; background: linear-gradient(90deg, var(--accent1), var(--accent2)); filter: drop-shadow(0 0 8px rgba(6,182,212,.35)); animation: pulseGlow 2.4s ease-in-out infinite; }
            @keyframes pulseGlow { 0%,100%{opacity:.5} 50%{opacity:1} }
            .btn-primary button { background: linear-gradient(90deg, var(--accent1), var(--accent2)); color:#fff; border-radius:999px; box-shadow: 0 6px 18px rgba(99,102,241,.35); }
            .btn-secondary button { background: transparent; color: var(--text); border: 1px solid var(--border); border-radius:999px; }
            a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 3px solid rgba(6,182,212,.45) !important; outline-offset: 1px; }
        </style>
        """,
        unsafe_allow_html=True,
)

# Hero header
st.markdown(
        """
        <section class="hero" aria-label="Hero">
            <h1 class="hero-title">vidéo into Advanced RAG</h1>
            <div class="hero-underline" aria-hidden="true"></div>
            <p class="muted">Transcribe • Chunk • Retrieve • Generate — entièrement en local</p>
        </section>
        """,
        unsafe_allow_html=True,
)

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
    task_id = st.text_input("Video title", value="video_1")
    ingest_btn = st.button("Start ingestion")
    st.markdown("---")
    if st.session_state.get("current_video_title"):
        st.markdown(f"<div class='sidebar-info'>Current video: <b>{st.session_state['current_video_title']}</b></div>", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    # instantiate lazily but cached across reruns
    return CombinedRAGService()


# We'll create the service lazily when a button is pressed to avoid model downloads on page load.
svc = None

# Main layout with a right-side log panel (2/3) and left content (1/3)
col_main, col_logs = st.columns([1, 2])
with col_logs:
    # Retrieval evidence panel (Top-k)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Retrieval evidence (Top-k)")
    last_prov = st.session_state.get("last_prov", [])
    if last_prov:
        for p in last_prov[:10]:
            cid = p.get('metadata',{}).get('chunk_id','')
            sim = p.get('score', None)
            fa = p.get('faiss_similarity', None)
            cl = p.get('cluster_id', None)
            rk = p.get('rank', None)
            meta = []
            if rk is not None: meta.append(f"r={rk}")
            if isinstance(sim, (int,float)): meta.append(f"sim={sim:.2f}")
            if isinstance(fa, (int,float)): meta.append(f"faiss={fa:.2f}")
            if cl is not None: meta.append(f"cl={cl}")
            st.write(f"- {cid} ({', '.join(meta)})")
    else:
        st.caption("No retrieval results yet. Ask a question after ingestion.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Logs panel
    st.markdown("### Logs")
    log_box = st.empty()
logs: list[str] = []


def append_log(msg: str):
    from datetime import datetime

    ts = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{ts}] {msg}")
    # keep only last 200 lines
    display = "\n".join(logs[-200:])
    # show logs in a monospace block
    log_box.code(display, language="")


if ingest_btn:
    # instantiate service here to avoid eager model loads
    try:
        svc = get_service()
    except Exception as e:
        append_log(f"[error] Failed to initialize RAG service: {e}")
        st.error("Failed to initialize RAG service; check logs in the sidebar.")

    if svc:
        append_log(f"[ui] Starting ingestion for {youtube_url} -> title {task_id}")
        with st.spinner("Ingesting video — see logs on the right for detailed progress..."):
            res = svc.ingest_from_youtube(youtube_url, task_id=task_id, progress_callback=append_log)

        if res.get("status") == "completed":
            append_log("[ui] Ingestion completed successfully")
            st.success("Ingestion completed")
            # Remember current video title and show in sidebar
            st.session_state["current_video_title"] = task_id
            with st.sidebar:
                st.markdown(f"<div class='sidebar-info'>Current video: <b>{task_id}</b></div>", unsafe_allow_html=True)
            with col_logs:
                st.markdown("<div class='card'>**Index result**</div>", unsafe_allow_html=True)
                st.json(res.get("index_result"))
            # Quick summary lines for clusters and chunks
            idx = res.get("index_result") or {}
            total_chunks = idx.get("total_chunks")
            n_clusters = idx.get("clusters")
            silh = idx.get("silhouette_score")
            append_log(f"[index] total_chunks={total_chunks}, clusters={n_clusters}, silhouette={silh}")
        else:
            append_log(f"[ui] Ingestion failed: {res}")
            st.error(f"Ingestion failed: {res}")

st.markdown("---")

with col_main:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Chat")

    # Initialize session messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {role, content}

    # Render history
    for m in st.session_state["messages"]:
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(m.get("content", ""))

    user_msg = st.chat_input("Message")
    if user_msg:
        # ensure service exists
        if svc is None:
            try:
                svc = get_service()
            except Exception as e:
                append_log(f"[error] Failed to initialize RAG service: {e}")
                st.error("Failed to initialize RAG service; check logs.")

        if svc is not None:
            # Add user message to history and show
            st.session_state["messages"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.spinner("Retrieving and answering..."):
                # Always use RAG with memory + corrective enabled
                out = svc.answer_query(
                    task_id=task_id,
                    query=user_msg,
                    k=5,
                    similarity_threshold=0.0,
                    use_memory=True,
                    progress_callback=append_log,
                    enable_corrective=True,
                )

            ans = out.get("answer", "")
            prov = out.get("provenance", [])
            with st.chat_message("assistant"):
                st.caption("Grounded (RAG)")
                st.markdown(ans)
                # Hide provenance in the chat UI per request; logs will still show retrieval diagnostics.

            # Append assistant message to history
            st.session_state["messages"].append({"role": "assistant", "content": ans})

            # Detailed retrieval diagnostics to log panel
            if prov:
                # keep last provenance for the evidence panel
                st.session_state["last_prov"] = prov
                append_log("[retrieval] Top results (rank, sim, grade, faiss, cluster, chunk_id):")
                for p in prov[:10]:
                    line = (
                        f"  - r={p.get('rank')}, sim={p.get('score'):.2f}, "
                        f"grade={p.get('grade',0.0):.2f} faiss={p.get('faiss_similarity',0.0):.2f}, "
                        f"cl={p.get('cluster_id','-')} id={(p.get('metadata',{}).get('chunk_id',''))}"
                    )
                    append_log(line)

    st.markdown("</div>", unsafe_allow_html=True)
