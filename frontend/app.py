# frontend/app.py
"""
SmartDoc AI v5 — Workspace Edition
• Workspace management (create / switch / delete)
• Each workspace = isolated ChromaDB collection
• Combined multi-doc search within a workspace
• Dual LLM (Llama 3.3 70B + Nemotron 70B) with 3-tier fallback
• LangChain LCEL streaming
• All previous features preserved
"""

import json
import os
import sys
import time
import base64
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.logging_config import setup_logging
setup_logging()

import streamlit as st
import streamlit.components.v1 as components

from backend.ingestion_pipeline import DocumentIngestion
from backend.batch_processor    import BatchProcessor
from backend.search_engine      import SearchEngine
from backend.session_manager    import SessionManager
from utils.error_handler        import (
    validate_uploaded_file, validate_query,
    classify_openai_error, check_system_health,
)
import logging
logger = logging.getLogger("smartdocs.app")

try:
    from config import (
        AVAILABLE_MODELS, PRIMARY_MODEL, DEFAULT_WORKSPACE_NAME,
    )
except ImportError:
    PRIMARY_MODEL          = "meta/llama-3.3-70b-instruct"
    DEFAULT_WORKSPACE_NAME = "Default"
    AVAILABLE_MODELS = {
        "⚡ Llama 3.3 70B  (Fast & General)":             "meta/llama-3.3-70b-instruct",
        "🎯 Nemotron 70B  (NVIDIA · Grounded Accuracy)":  "nvidia/llama-3.1-nemotron-70b-instruct",
    }

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Dark theme (permanent) ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{font-family:Inter,system-ui,sans-serif!important;}
:root{
  --bg:#0B0F14;--surf:#121720;--bord:#1E2533;
  --t1:#E6EAF0;--t2:#9AA4B2;--tm:#6B7280;
  --ac:#6366F1;--ok:#22C55E;--err:#EF4444;--warn:#F59E0B;
}
.stApp{background:var(--bg)!important;color:var(--t1)!important;}
section[data-testid="stSidebar"]{background:var(--bg)!important;
  border-right:1px solid var(--bord);}
section[data-testid="stSidebar"]>div{padding:1rem .9rem;}
.stFileUploader{background:var(--surf)!important;
  border:2px dashed var(--bord)!important;border-radius:12px!important;}
.stFileUploader label,.stFileUploader div,
.stFileUploader p,.stFileUploader span{
  color:var(--t1)!important;background:transparent!important;}
.stFileUploader [data-testid="stFileUploaderDropzone"]{
  background:var(--surf)!important;border-color:var(--bord)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent;
  border-bottom:1px solid var(--bord);}
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}
.stTabs [data-baseweb="tab"]{background:transparent;border:none;
  border-bottom:2px solid transparent;color:var(--t2);font-size:14px;
  font-weight:500;padding:.65rem 1.25rem;transition:all .15s;}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color:var(--t1);border-bottom:2px solid var(--ac);}
.stButton>button{background:var(--ac);color:#fff;border-radius:10px;
  border:none;font-weight:600;font-size:13px;padding:.55rem 1.1rem;
  transition:all .2s;box-shadow:0 2px 8px rgba(99,102,241,.3);}
.stButton>button:hover{background:#4F46E5;transform:translateY(-1px);}
.stTextArea textarea,.stTextInput input{background:var(--surf)!important;
  border:1px solid var(--bord)!important;border-radius:8px!important;
  color:var(--t1)!important;font-size:14px!important;}
.stSelectbox>div>div,.stSelectbox label{
  background:var(--surf)!important;color:var(--t1)!important;
  border-color:var(--bord)!important;}
.stSelectbox [data-baseweb="select"]>div{
  background:var(--surf)!important;border-color:var(--bord)!important;}
.stMetric label{font-size:11px;color:var(--t2);font-weight:500;
  text-transform:uppercase;letter-spacing:.04em;}
.stMetric [data-testid="stMetricValue"]{font-size:26px;font-weight:700;
  color:var(--ac);}
.stExpander{border:1px solid var(--bord)!important;border-radius:10px!important;
  background:var(--surf)!important;margin-bottom:.5rem;}
.stExpander summary{color:var(--t1)!important;font-weight:500;font-size:14px;}
.stProgress>div>div>div{background:var(--ac)!important;border-radius:6px!important;}
.stSuccess,.stError,.stWarning,.stInfo{border-radius:10px!important;}
.user-msg{background:rgba(99,102,241,.15);border-radius:18px 18px 4px 18px;
  padding:.85rem 1.1rem;margin:.4rem 0 .4rem auto;max-width:74%;
  font-size:14px;line-height:1.65;color:var(--t1);}
.ai-msg{background:var(--surf);border:1px solid var(--bord);
  border-radius:4px 18px 18px 18px;padding:.85rem 1.1rem;
  margin:.4rem auto .4rem 0;max-width:74%;
  font-size:14px;line-height:1.7;color:var(--t1);}
.msg-meta{font-size:11px;color:var(--tm);margin-bottom:.3rem;font-weight:600;}
.cite{display:inline-block;background:rgba(99,102,241,.18);
  border:1px solid rgba(99,102,241,.35);border-radius:20px;
  padding:.18rem .65rem;font-size:11px;color:var(--ac);font-weight:600;margin:2px;}
.fallback-badge{display:inline-block;background:rgba(245,158,11,.15);
  border:1px solid rgba(245,158,11,.4);border-radius:20px;
  padding:.18rem .65rem;font-size:11px;color:#F59E0B;font-weight:600;margin:2px;}
.model-badge{display:inline-block;background:rgba(99,102,241,.1);
  border:1px solid rgba(99,102,241,.25);border-radius:20px;
  padding:.15rem .6rem;font-size:10px;color:var(--ac);font-weight:600;margin:2px;}
.ws-badge{display:inline-block;background:rgba(34,197,94,.12);
  border:1px solid rgba(34,197,94,.3);border-radius:20px;
  padding:.15rem .6rem;font-size:10px;color:#22C55E;font-weight:600;margin:2px;}
.ws-card{background:var(--surf);border:1px solid var(--bord);
  border-radius:10px;padding:.75rem 1rem;margin:.4rem 0;}
.ws-card.active{border-color:var(--ac);background:rgba(99,102,241,.06)!important;}
.ws-name{font-size:13px;font-weight:600;color:var(--t1);}
.ws-meta{font-size:11px;color:var(--tm);margin-top:.2rem;}
.src-wrap{background:var(--surf);border:1px solid var(--bord);
  border-radius:8px;padding:.75rem 1rem;margin:.35rem 0;}
.src-lbl{font-size:12px;color:var(--t1);font-weight:500;margin-bottom:.3rem;}
.src-track{background:var(--bord);border-radius:4px;height:7px;width:100%;overflow:hidden;}
.src-fill{height:7px;border-radius:4px;
  background:linear-gradient(90deg,var(--ac),#818CF8);}
.src-pct{font-size:10px;color:var(--tm);margin-top:.2rem;}
.doc-card{background:var(--surf);border:1px solid var(--bord);
  border-radius:10px;padding:.75rem 1rem;margin:.4rem 0;}
.doc-name{font-size:13px;font-weight:600;color:var(--t1);}
.doc-meta{font-size:11px;color:var(--tm);margin-top:.2rem;}
.brand{font-size:19px;font-weight:700;color:var(--t1);letter-spacing:-.02em;}
.brand-tag{font-size:11px;color:var(--tm);margin-bottom:.75rem;}
.card{background:var(--surf);border:1px solid var(--bord);border-radius:12px;
  padding:1rem 1.25rem;margin-bottom:.75rem;
  box-shadow:0 2px 8px rgba(0,0,0,.15);}
.hr{border:none;border-top:1px solid var(--bord);margin:.75rem 0;}
.perf-metric{display:flex;flex-direction:column;align-items:center;
  background:var(--surf);border:1px solid var(--bord);border-radius:10px;
  padding:.75rem;text-align:center;}
.perf-val{font-size:22px;font-weight:700;color:var(--ac);}
.perf-lbl{font-size:10px;color:var(--tm);text-transform:uppercase;
  letter-spacing:.06em;margin-top:.2rem;}
</style>""", unsafe_allow_html=True)


# ── Session bootstrap ─────────────────────────────────────────────────────────
sm = SessionManager()

if "session" not in st.session_state:
    st.session_state.session         = sm.create_session()
    active_ws = DEFAULT_WORKSPACE_NAME
    st.session_state.pipeline        = DocumentIngestion(workspace=active_ws)
    st.session_state.search_engine   = SearchEngine(
        model=PRIMARY_MODEL, workspace=active_ws
    )
    st.session_state.batch_processor = BatchProcessor()
    st.session_state.query_cache     = []
    st.session_state.search_results  = []
    st.session_state.results_page    = 0
    st.session_state.pdf_b64         = {}
    st.session_state.active_model    = PRIMARY_MODEL
    st.session_state.chroma_synced   = False

sess = st.session_state.session
inject_css()

RESULTS_PER_PAGE = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def cached_health():
    return check_system_health()

def _ts(m):
    ts = m.get("timestamp", "")
    if isinstance(ts, datetime): return ts.strftime("%H:%M")
    if isinstance(ts, str) and "T" in ts: return ts[11:16]
    return str(ts)[:5]

def _active_ws() -> str:
    return sm.get_active_workspace(sess)

def _ws_docs() -> list:
    return sm.get_workspace_docs(sess)

def _db_count() -> int:
    try:
        return st.session_state.pipeline.vector_db \
            .get_collection_stats().get("count", 0)
    except Exception:
        return 0

def _model_label(model_id: str) -> str:
    for label, mid in AVAILABLE_MODELS.items():
        if mid == model_id:
            return label
    return model_id

def _encode_pdf(f) -> str:
    if hasattr(f, "getvalue"):
        f.seek(0)
        return base64.b64encode(f.getvalue()).decode()
    with open(f, "rb") as fh:
        return base64.b64encode(fh.read()).decode()

def _render_badges(cites: list):
    if not cites: return
    html = '<div style="margin:.4rem 0 .6rem;">'
    for c in cites:
        html += f'<span class="cite">📄 {c["file"]} · p.{c["page"]}</span> '
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _render_bars(dist: dict):
    if not dist or not dist.get("percentages"): return
    st.caption("**Document contribution:**")
    for src, pct in dist["percentages"].items():
        short = src[-30:] if len(src) > 30 else src
        st.markdown(f"""
<div class="src-wrap">
  <div class="src-lbl">📎 {short}</div>
  <div class="src-track">
    <div class="src-fill" style="width:{int(pct)}%;"></div>
  </div>
  <div class="src-pct">{pct}% of retrieved chunks</div>
</div>""", unsafe_allow_html=True)

def _switch_workspace(new_ws: str):
    """Switch all stateful components to a new workspace."""
    sm.set_active_workspace(sess, new_ws)
    st.session_state.pipeline.switch_workspace(new_ws)
    st.session_state.search_engine.switch_workspace(new_ws)
    # Clear chat — context from old workspace doesn't apply
    sess["chat_history"]             = []
    st.session_state.search_results  = []
    st.session_state.query_cache     = []
    st.session_state.results_page    = 0
    st.session_state.chroma_synced   = False
    logger.info(f"UI switched to workspace '{new_ws}'")

def _delete_document(doc_name: str) -> int:
    vdb = st.session_state.pipeline.vector_db
    ids = vdb.get_source_file_ids(doc_name)
    if ids:
        vdb.delete_documents(ids)
    sm.remove_doc_from_workspace(sess, _active_ws(), doc_name)
    st.session_state.pdf_b64.pop(doc_name, None)
    return len(ids)

def _sync_docs_from_chroma():
    """Populate workspace doc list from ChromaDB on session start."""
    try:
        docs_in_chroma = st.session_state.pipeline.vector_db \
            .get_workspace_docs(_active_ws())
        existing = {d["name"] for d in _ws_docs()}
        for src in docs_in_chroma:
            if src not in existing:
                sm.add_doc_to_workspace(sess, _active_ws(), {
                    "name": src, "pages": 0, "chunks": 0, "summary": "",
                })
    except Exception as e:
        logger.warning(f"ChromaDB sync skipped: {e}")

if not st.session_state.chroma_synced:
    _sync_docs_from_chroma()
    st.session_state.chroma_synced = True


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="brand">📚 SmartDoc AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-tag">Intelligent Document Q&A · v5.0</div>',
        unsafe_allow_html=True,
    )


    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    # ── Workspace Panel ────────────────────────────────────────────────────────
    with st.expander("🗂 Workspaces", expanded=True):
        active_ws   = _active_ws()
        ws_names    = sm.list_workspace_names(sess)

        # Workspace cards
        for ws in ws_names:
            ws_docs  = sm.get_workspace_docs(sess, ws)
            is_active = ws == active_ws
            card_cls  = "ws-card active" if is_active else "ws-card"
            st.markdown(
                f'<div class="{card_cls}">'
                f'<div class="ws-name">'
                f'{"🟢 " if is_active else "⚪ "}{ws}</div>'
                f'<div class="ws-meta">{len(ws_docs)} doc(s)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if not is_active:
                col_sw, col_del = st.columns([3, 1])
                with col_sw:
                    if st.button(
                        f"Switch → {ws[:18]}", key=f"sw_{ws}",
                        use_container_width=True,
                    ):
                        _switch_workspace(ws)
                        st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_ws_{ws}",
                                 help=f"Delete workspace '{ws}' and all its chunks"):
                        try:
                            st.session_state.pipeline.vector_db \
                                .delete_workspace(ws)
                            del sess["workspaces"][ws]
                            st.success(f"Workspace '{ws}' deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        # Create new workspace
        new_ws_name = st.text_input(
            "New workspace name",
            placeholder="e.g. AI Research, Harry Potter…",
            key="new_ws_input",
            label_visibility="collapsed",
        )
        if st.button("➕ Create Workspace", use_container_width=True,
                     key="create_ws_btn"):
            name = new_ws_name.strip()
            if not name:
                st.warning("Enter a workspace name.")
            elif name in ws_names:
                st.warning(f"'{name}' already exists.")
            else:
                sm.create_workspace(sess, name)
                st.success(f"Created '{name}'! Switch to start uploading.")
                st.rerun()

    # ── LLM Selector ──────────────────────────────────────────────────────────
    with st.expander("🤖 AI Model", expanded=False):
        model_labels  = list(AVAILABLE_MODELS.keys())
        current_label = _model_label(st.session_state.active_model)
        default_idx   = (model_labels.index(current_label)
                         if current_label in model_labels else 0)
        selected_label = st.selectbox(
            "Model", model_labels, index=default_idx,
            key="model_selector", label_visibility="collapsed",
            help="Switch between Llama 3.3 70B and Nemotron 70B",
        )
        selected_model = AVAILABLE_MODELS[selected_label]
        if selected_model != st.session_state.active_model:
            st.session_state.search_engine.switch_model(selected_model)
            st.session_state.active_model = selected_model
            st.success(f"Switched to {selected_label.split('(')[0].strip()}")
            st.rerun()
        if "Nemotron" in selected_label:
            st.info("🎯 **Nemotron 70B** — RLHF-tuned for grounded accuracy.")
        else:
            st.info("⚡ **Llama 3.3 70B** — fast, accurate, great for most tasks.")

    # ── Upload PDFs ───────────────────────────────────────────────────────────
    with st.expander("📤 Upload PDFs", expanded=True):
        st.caption(f"Uploading into: **{active_ws}**")
        uploaded_files = st.file_uploader(
            "Choose PDFs", type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.pdf_b64:
                    try:
                        st.session_state.pdf_b64[f.name] = _encode_pdf(f)
                    except Exception:
                        pass

    # ── PDF Viewer ─────────────────────────────────────────────────────────────
    all_names = list(st.session_state.pdf_b64.keys())
    if all_names:
        with st.expander("🔍 PDF Viewer", expanded=False):
            chosen_pdf = st.selectbox(
                "Select", all_names, label_visibility="collapsed",
            )
            if chosen_pdf:
                b64 = st.session_state.pdf_b64.get(chosen_pdf)
                if b64:
                    components.html(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="480px" '
                        f'style="border:none;border-radius:10px;"></iframe>',
                        height=490,
                    )

    # ── Indexed docs in active workspace ──────────────────────────────────────
    docs = _ws_docs()
    with st.expander(
        f"📄 Docs in '{active_ws}' ({len(docs)})", expanded=False
    ):
        if docs:
            for doc in docs:
                col_info, col_del = st.columns([5, 1])
                with col_info:
                    summary = doc.get("summary", "")
                    st.markdown(
                        f'<div class="doc-card">'
                        f'<div class="doc-name">📄 {doc["name"]}</div>'
                        f'<div class="doc-meta">'
                        f'{doc.get("pages",0)} pages · '
                        f'{doc.get("chunks",0)} chunks</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if summary:
                        st.markdown(
                            f'<div style="font-size:11px;color:var(--tm);'
                            f'padding:.2rem .5rem .3rem;line-height:1.5;">'
                            f'💡 {summary[:180]}'
                            f'{"…" if len(summary)>180 else ""}</div>',
                            unsafe_allow_html=True,
                        )
                with col_del:
                    if st.button("🗑", key=f"del_{active_ws}_{doc['name']}",
                                 help=f"Remove {doc['name']}"):
                        try:
                            n = _delete_document(doc["name"])
                            st.success(f"Removed {n} chunks")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        else:
            st.caption("_No documents in this workspace yet_")

    # ── Processing Settings ────────────────────────────────────────────────────
    with st.expander("⚙️ Processing Settings"):
        clean_text       = st.checkbox("Clean extracted text", True)
        replace_existing = st.checkbox("Replace if already indexed", False)
        gen_summary      = st.checkbox("Auto-generate AI summary", True)
        chunk_method     = st.selectbox("Chunking strategy", ["tokens","sentences"])
        chunk_size       = st.slider("Chunk size (tokens)", 200, 2000,
                                     sess["preferences"].get("chunk_size", 500), 100)
        overlap          = st.slider("Chunk overlap", 0, 400,
                                     sess["preferences"].get("overlap", 100), 50)
        top_k            = st.slider("Top-K results", 1, 15, 12)
        threshold        = st.slider("Min similarity", 0.0, 1.0, 0.25, 0.05)
        sess["preferences"].update({"chunk_size": chunk_size, "overlap": overlap})

    # ── Session Management ─────────────────────────────────────────────────────
    with st.expander("💾 Session Management"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save", use_container_width=True):
                sm.save_session_state(sess)
                st.success("Saved!")
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.session         = sm.create_session()
                st.session_state.search_results  = []
                st.session_state.query_cache     = []
                st.session_state.pdf_b64         = {}
                st.session_state.chroma_synced   = False
                st.rerun()
        sessions = sm.list_sessions()
        if sessions:
            sid_map = {s["session_id"][:8]: s["session_id"] for s in sessions}
            chosen  = st.selectbox("Restore session",
                                   ["—"] + list(sid_map.keys()))
            if chosen != "—" and st.button("↩ Load"):
                loaded = sm.load_session_state(sid_map[chosen])
                if loaded:
                    st.session_state.session       = loaded
                    st.session_state.chroma_synced = False
                    st.success("Restored!")
                    st.rerun()
        stats = sm.get_statistics(sess)
        st.markdown(f"""
<div style="font-size:12px;color:var(--tm);line-height:2;">
  Queries: <b>{stats['query_count']}</b> &nbsp;|&nbsp;
  Workspaces: <b>{stats['workspace_count']}</b><br>
  Docs: <b>{stats['documents']}</b> &nbsp;|&nbsp;
  Avg: <b>{stats['avg_response_s']}s</b>
</div>""", unsafe_allow_html=True)

    # ── Health ─────────────────────────────────────────────────────────────────
    with st.expander("🔍 System Health"):
        if st.button("Run Check", use_container_width=True):
            cached_health.clear()
            st.session_state.health_result = cached_health()
        h = st.session_state.get("health_result")
        if h:
            for k, v in h.items():
                if k == "overall": continue
                st.markdown(f"{'✅' if v['ok'] else '❌'} **{k}:** {v['message']}")

    with st.expander("🗃 Workspace Diagnostics"):
        st.caption(f"What's actually in **{active_ws}** (from ChromaDB)")
        if st.button("Refresh", key="diag_refresh", use_container_width=True):
            st.session_state.diag_stats = None
        if "diag_stats" not in st.session_state or st.session_state.diag_stats is None:
            try:
                st.session_state.diag_stats = (
                    st.session_state.pipeline.vector_db.get_workspace_doc_stats()
                )
            except Exception as e:
                st.session_state.diag_stats = []
                st.error(f"Diagnostics failed: {e}")
        diag = st.session_state.diag_stats or []
        if diag:
            total_chunks = sum(d["chunks"] for d in diag)
            st.markdown(
                f"**{len(diag)} document(s) · {total_chunks} total chunks**"
            )
            for d in diag:
                bar_pct = int(d["chunks"] / max(total_chunks, 1) * 100)
                st.markdown(f"""
<div class="src-wrap">
  <div class="src-lbl">📄 {d['name']}</div>
  <div class="src-track">
    <div class="src-fill" style="width:{bar_pct}%;"></div>
  </div>
  <div class="src-pct">
    {d['chunks']} chunks · {d['pages']} pages
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.caption("No documents indexed in this workspace yet.")

    with st.expander("ℹ️ About"):
        st.markdown(f"""
**Stack:** LangChain · NVIDIA NIM · ChromaDB · Streamlit

**Active workspace:** `{active_ws}`
**Active model:** `{st.session_state.active_model}`

**Workspace isolation:**
Each workspace = separate ChromaDB collection.
Harry Potter and your Project Report never contaminate each other.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# STATS STRIP
# ═══════════════════════════════════════════════════════════════════════════════
docs     = _ws_docs()
db_count = _db_count()
times    = sess.get("response_times", [])
avg_rt   = f"{sum(times)/len(times):.1f}s" if times else "—"

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("🗂 Workspace",    active_ws[:12] + ("…" if len(active_ws)>12 else ""))
mc2.metric("📄 Documents",   len(docs))
mc3.metric("🗄 Chunks",      db_count)
mc4.metric("💬 Queries",     sess.get("query_count", 0))
mc5.metric("⚡ Avg Response", avg_rt)
st.markdown('<hr class="hr">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_upload, tab_ask, tab_history, tab_export = st.tabs(
    ["📤  Upload Documents", "💬  Ask Questions", "🕑  History", "📥  Export"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(f"## Upload into: **{active_ws}**")
    st.caption(
        "Documents uploaded here will be indexed into the "
        f"**{active_ws}** workspace only. "
        "Switch workspaces in the sidebar to keep topics separate."
    )

    if uploaded_files:
        valid, invalid = [], []
        for f in uploaded_files:
            try:
                validate_uploaded_file(f)
                valid.append(f)
            except Exception as ve:
                invalid.append((f.name, str(ve)))

        for name, msg in invalid:
            st.error(f"❌ **{name}**: {msg}")

        if valid:
            import pandas as pd
            st.dataframe(
                pd.DataFrame([{
                    "File": f.name,
                    "Size (MB)": f"{f.size/1024/1024:.2f}",
                    "Workspace": active_ws,
                    "Status": "✅ Ready",
                } for f in valid]),
                use_container_width=True, hide_index=True,
            )

        if valid and st.button("⚡  Index Documents", type="primary"):
            prog  = st.progress(0, text="Preparing…")
            stage = st.empty()

            def prog_cb(done, tot, fname):
                frac = done / tot
                prog.progress(frac, text=f"{fname} ({done}/{tot})")
                stage.caption(f"Embedding… {int(frac*100)}%")

            try:
                os.makedirs("uploads", exist_ok=True)
                for f in valid:
                    f.seek(0)
                    with open(f"uploads/{f.name}", "wb") as out:
                        out.write(f.getbuffer())

                t_start = time.time()
                summary_dict = st.session_state.batch_processor.process_files(
                    valid,
                    workspace=active_ws,
                    clean_text=clean_text,
                    chunk_strategy=chunk_method,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    replace_existing=replace_existing,
                    generate_summary=gen_summary,
                    progress_callback=prog_cb,
                )
                elapsed = round(time.time() - t_start, 2)
                prog.progress(1.0, text="Done! ✅")
                stage.empty()

                # Perf metrics
                pm1, pm2, pm3, pm4 = st.columns(4)
                for col, val, lbl in [
                    (pm1, f"{summary_dict['successful']}/{len(valid)}", "Files Indexed"),
                    (pm2, summary_dict["total_chunks"],                  "Total Chunks"),
                    (pm3, summary_dict.get("chunks_per_sec","—"),        "Chunks / Sec"),
                    (pm4, f"{elapsed}s",                                 "Total Time"),
                ]:
                    col.markdown(
                        f'<div class="perf-metric">'
                        f'<div class="perf-val">{val}</div>'
                        f'<div class="perf-lbl">{lbl}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )

                # Per-file results
                for r in summary_dict["results"]:
                    if r.get("success"):
                        if r.get("already_indexed"):
                            st.info(
                                f"⏭ **{r['file_name']}** already indexed in "
                                f"**{active_ws}** ({r.get('existing_chunks',0)} chunks). "
                                "Enable **Replace if already indexed** to re-process."
                            )
                            existing_names = [d["name"] for d in _ws_docs()]
                            if r["file_name"] not in existing_names:
                                sm.add_doc_to_workspace(sess, active_ws, {
                                    "name":    r["file_name"],
                                    "pages":   0,
                                    "chunks":  r.get("existing_chunks", 0),
                                    "summary": "",
                                })
                        else:
                            sm.add_doc_to_workspace(sess, active_ws, {
                                "name":    r["file_name"],
                                "pages":   r.get("pages", 0),
                                "chunks":  r.get("chunks", 0),
                                "summary": r.get("summary", ""),
                            })
                            st.success(
                                f"✅ **{r['file_name']}** · "
                                f"{r.get('pages',0)} pages · "
                                f"{r.get('chunks',0)} chunks · "
                                f"{r.get('chunks_per_sec','—')} chunks/s · "
                                f"workspace: **{active_ws}**"
                            )
                            if r.get("summary"):
                                with st.expander(
                                    f"💡 AI Summary: {r['file_name']}"
                                ):
                                    st.write(r["summary"])
                    else:
                        st.error(f"❌ {r['file_name']}: {r.get('error','unknown')}")

                sm.save_session_state(sess)
                st.info("✨ Switch to **Ask Questions** to query this workspace.")
                st.rerun()

            except Exception as e:
                prog.empty(); stage.empty()
                logger.error(f"Indexing error: {e}")
                st.error(f"❌ {str(e)[:200]}")
    else:
        st.markdown(f"""
<div class="card">
  <h4 style="margin:0 0 .5rem;font-size:15px;">
      📋 Active workspace: <span style="color:var(--ac)">{active_ws}</span>
  </h4>
  <p style="font-size:13px;color:var(--t2);margin:.5rem 0;">
      All documents uploaded here are indexed into
      <b>{active_ws}</b> only — completely isolated from other workspaces.
  </p>
  <hr style="border:none;border-top:1px solid var(--bord);margin:.75rem 0;">
  <p style="font-size:13px;color:var(--t2);margin:0;">
      💡 <b>Tip:</b> Create separate workspaces for unrelated topics
      (e.g. "AI Research", "Harry Potter", "Legal Docs") to prevent
      cross-contamination in search results.
  </p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ASK QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown(f"## Ask Questions — **{active_ws}**")

    chroma_count = _db_count()
    if not docs and chroma_count == 0:
        st.info(
            f"📤 No documents in **{active_ws}** yet. "
            "Upload PDFs in the Upload tab, or switch workspaces."
        )
    else:
        chat      = sess.get("chat_history", [])
        bookmarks = sess.get("bookmarks", [])

        # ── Chat history ───────────────────────────────────────────────────────
        for i, msg in enumerate(chat):
            is_bm = i in bookmarks
            ts    = _ts(msg)
            if msg["role"] == "user":
                bm_col, _ = st.columns([1, 14])
                with bm_col:
                    if st.button("⭐" if is_bm else "☆", key=f"bm{i}"):
                        if is_bm: sm.remove_bookmark(sess, i)
                        else:     sm.add_bookmark(sess, i)
                        st.rerun()
                st.markdown(
                    f'<div class="msg-meta">You · {ts}</div>'
                    f'<div class="user-msg">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                model_used  = msg.get("model_used", "")
                fallback    = msg.get("fallback_used", False)
                fallback_t  = msg.get("fallback_tier", 1)
                msg_ws      = msg.get("workspace", active_ws)
                model_short = model_used.split("/")[-1] if model_used else ""

                if fallback and fallback_t == 2:
                    fb_html = (
                        f'<span class="fallback-badge">'
                        f'⚡ auto-switched → {model_short}</span>'
                    )
                elif fallback and fallback_t == 3:
                    fb_html = (
                        '<span class="fallback-badge">'
                        '⚠️ both models failed</span>'
                    )
                else:
                    fb_html = ""

                st.markdown(
                    f'<div class="msg-meta">'
                    f'SmartDoc AI · {ts} '
                    f'<span class="model-badge">{model_short}</span>'
                    f'<span class="ws-badge">🗂 {msg_ws}</span>'
                    f'{fb_html}</div>'
                    f'<div class="ai-msg">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                _render_badges(msg.get("citations", []))
                dist = msg.get("source_distribution")
                if dist and len(dist.get("percentages", {})) >= 1:
                    with st.expander("📊 Source Contribution"):
                        _render_bars(dist)

        # ── Paginated chunks ───────────────────────────────────────────────────
        raw = st.session_state.search_results
        if raw:
            page  = st.session_state.results_page
            start = page * RESULTS_PER_PAGE
            shown = raw[start:start + RESULTS_PER_PAGE]
            with st.expander(
                f"🔍 Retrieved Chunks "
                f"({min(start+RESULTS_PER_PAGE,len(raw))}/{len(raw)})"
            ):
                for r in shown:
                    st.markdown(
                        f"**{r.get('source','?')} · p.{r.get('page','?')}** "
                        f"*(score: {r.get('relevance_score',0):.3f})*"
                    )
                    st.caption(r.get("text","")[:300]+"…")
                    st.divider()
            if start + RESULTS_PER_PAGE < len(raw):
                if st.button("Load More ↓"):
                    st.session_state.results_page += 1
                    st.rerun()

        st.markdown('<hr class="hr">', unsafe_allow_html=True)

        # ── Document filter (within this workspace) ────────────────────────────
        if len(docs) > 1:
            with st.expander("🗂 Filter by Document (within workspace)", expanded=False):
                selected_docs = st.multiselect(
                    "Search only in these docs (empty = all docs in workspace)",
                    options=[d["name"] for d in docs],
                    default=[],
                    key="doc_filter",
                    help=(
                        "All docs here are already in the same workspace. "
                        "Use this to further narrow to 1-2 specific documents."
                    ),
                )
                if selected_docs:
                    st.caption(f"🔍 Filtering to: **{', '.join(selected_docs)}**")
                else:
                    st.caption(
                        f"🔍 Searching all **{len(docs)}** docs in **{active_ws}**"
                    )
        else:
            selected_docs = []

        doc_filter = None
        if selected_docs:
            doc_filter = (
                {"source_file": selected_docs[0]}
                if len(selected_docs) == 1
                else {"source_file": {"$in": selected_docs}}
            )

        # ── Input ──────────────────────────────────────────────────────────────
        q_input = st.text_area(
            "question_input",
            placeholder=f"Ask a question about documents in '{active_ws}'…",
            height=85,
            label_visibility="collapsed",
            key="q_input",
        )
        c_send, c_clear = st.columns([3, 1])
        with c_send:
            send = st.button("Send ✈", type="primary", use_container_width=True)
        with c_clear:
            if st.button("Clear 🗑", use_container_width=True):
                sess["chat_history"]             = []
                st.session_state.search_results  = []
                st.session_state.results_page    = 0
                st.session_state.search_engine.qa_engine.clear_memory()
                st.rerun()

        # ── Handle send ────────────────────────────────────────────────────────
        if send:
            try:
                clean_q = validate_query(q_input)
            except Exception as ve:
                st.warning(str(ve))
                st.stop()

            cached = next(
                (c for c in st.session_state.query_cache if c["q"] == clean_q),
                None,
            )

            if cached:
                result     = cached["result"]
                elapsed    = cached["elapsed"]
                raw_chunks = cached["raw_chunks"]
                st.caption("⚡ Loaded from cache")
            else:
                try:
                    raw_chunks = st.session_state.search_engine.search_similar_chunks(
                        query=clean_q,
                        top_k=top_k,
                        similarity_threshold=threshold,
                        filters=doc_filter,
                    )
                except Exception as e:
                    st.error(classify_openai_error(e))
                    st.stop()

                if not raw_chunks:
                    filter_hint = (
                        f" in {', '.join(selected_docs)}"
                        if selected_docs else f" in workspace '{active_ws}'"
                    )
                    st.warning(
                        f"⚠️ No relevant chunks found{filter_hint}. "
                        "Try lowering **Min Similarity** in ⚙️ Processing Settings."
                    )
                    st.stop()

                ts_now = datetime.now().isoformat()
                sess["chat_history"].append({
                    "role": "user", "content": clean_q, "timestamp": ts_now,
                })

                answer_ph = st.empty()
                parts     = []
                t0        = time.time()

                try:
                    for token in st.session_state.search_engine.qa_engine.stream_answer(
                        clean_q, raw_chunks
                    ):
                        parts.append(token)
                        answer_ph.markdown(
                            f'<div class="msg-meta">SmartDoc AI · live</div>'
                            f'<div class="ai-msg">{"".join(parts)}▌</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    fallback_res = st.session_state.search_engine._raw_chunk_fallback(
                        clean_q, raw_chunks, str(e)
                    )
                    parts = [fallback_res["answer"]]

                raw_answer = "".join(parts)
                elapsed    = round(time.time() - t0, 2)
                answer_ph.empty()

                qa_eng    = st.session_state.search_engine.qa_engine
                _, answer = qa_eng.split_reasoning(raw_answer)
                qa_eng.store_conversation_turn(clean_q, answer)

                try:
                    cites = qa_eng.parse_citations(answer)
                except Exception:
                    cites = []
                try:
                    dist = st.session_state.search_engine \
                        .get_source_distribution(raw_chunks)
                except Exception:
                    dist = {}

                result = {
                    "answer":               answer,
                    "citations":            cites,
                    "quality_check_passed": qa_eng.validate_answer_quality(
                        answer, raw_chunks
                    ),
                    "model_used":           st.session_state.active_model,
                    "fallback_used":        False,
                    "fallback_tier":        1,
                    "workspace":            active_ws,
                }

                c_entry = {
                    "q": clean_q, "result": result,
                    "elapsed": elapsed, "raw_chunks": raw_chunks,
                }
                st.session_state.query_cache.append(c_entry)
                if len(st.session_state.query_cache) > 10:
                    st.session_state.query_cache.pop(0)

            if not cached:
                sess["chat_history"].append({
                    "role":               "assistant",
                    "content":            result["answer"],
                    "timestamp":          datetime.now().isoformat(),
                    "citations":          result.get("citations", []),
                    "model_used":         result.get("model_used", ""),
                    "fallback_used":      result.get("fallback_used", False),
                    "fallback_tier":      result.get("fallback_tier", 1),
                    "workspace":          result.get("workspace", active_ws),
                    "source_distribution": dist if not cached else {},
                })
                sess["query_count"]      += 1
                sess["response_times"].append(elapsed)
                st.session_state.search_results = raw_chunks
                st.session_state.results_page   = 0
                if sm.should_auto_save(sess["query_count"]):
                    sm.save_session_state(sess)

            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("## Conversation History")
    chat = sess.get("chat_history", [])
    bms  = sess.get("bookmarks", [])

    if not chat:
        st.info("No history yet.")
    else:
        if bms:
            st.markdown("### ⭐ Bookmarked")
            for idx in bms:
                if idx < len(chat):
                    m = chat[idx]
                    r = "**You**" if m["role"]=="user" else "**SmartDoc AI**"
                    st.markdown(f"{r} · _{_ts(m)}_")
                    st.info(
                        m["content"][:400]+("…" if len(m["content"])>400 else "")
                    )
            st.divider()

        st.markdown("### All Questions")
        import pandas as pd
        rows = [
            {
                "#":        i // 2 + 1,
                "Time":     _ts(m),
                "Workspace": m.get("workspace", active_ws) if m["role"]=="assistant" else "",
                "Question": m["content"][:90]+("…" if len(m["content"])>90 else ""),
                "⭐":        "⭐" if i in bms else "",
            }
            for i, m in enumerate(chat) if m["role"] == "user"
        ]
        if rows:
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("## Export & Download")
    if not sess.get("chat_history"):
        st.info("No conversation to export yet.")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        col_t, col_m, col_j = st.columns(3)
        with col_t:
            st.download_button(
                "📄 Plain Text",
                data=sm.export_as_text(sess).encode(),
                file_name=f"smartdoc_{stamp}.txt", mime="text/plain",
                use_container_width=True,
            )
        with col_m:
            md = sm.export_as_markdown(sess)
            st.download_button(
                "📝 Markdown",
                data=md.encode(),
                file_name=f"smartdoc_{stamp}.md", mime="text/markdown",
                use_container_width=True,
            )
        with col_j:
            st.download_button(
                "💾 Session JSON",
                data=json.dumps(sess, default=str, indent=2).encode(),
                file_name=f"smartdoc_{sess['session_id'][:8]}.json",
                mime="application/json", use_container_width=True,
            )
        with st.expander("Preview (Markdown)"):
            st.markdown(md[:3000]+("…" if len(md)>3000 else ""))