# config.py
"""
SmartDocs AI — Centralised Configuration
All tunable constants live here. Never hard-code values in individual files.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent

# ── NVIDIA NIM ────────────────────────────────────────────────────────────────
NVIDIA_BASE_URL          = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_API_KEY      = os.getenv("NVIDIA_CHAT_API_KEY")
NVIDIA_EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")

# ── Dual LLM Backends ─────────────────────────────────────────────────────────
PRIMARY_MODEL   = "meta/llama-3.3-70b-instruct"
SECONDARY_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

AVAILABLE_MODELS = {
    "⚡ Llama 3.3 70B  (Fast & General)":             "meta/llama-3.3-70b-instruct",
    "🎯 Nemotron 70B  (NVIDIA · Grounded Accuracy)":  "nvidia/llama-3.1-nemotron-70b-instruct",
}

CHAT_MODEL = PRIMARY_MODEL

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIMENSION = 1024

# ── Retry / rate-limit ────────────────────────────────────────────────────────
MAX_RETRIES             = 3
MAX_REQUESTS_PER_MINUTE = 60

# ── Workspace / Collection management ─────────────────────────────────────────
# Each workspace maps to a separate ChromaDB collection.
# Collection name format: "smartdocs_<workspace_slug>"
# e.g. workspace "AI Research" → collection "smartdocs_ai_research"
WORKSPACE_COLLECTION_PREFIX = "smartdocs_"
DEFAULT_WORKSPACE_NAME      = "Default"

# ── Vector database ───────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(ROOT_DIR / "data" / "chroma_db")

# ── Embedding cache ───────────────────────────────────────────────────────────
EMBEDDING_CACHE_FILE = ROOT_DIR / "data" / "embedding_cache.json"
MAX_CACHE_ENTRIES    = 10_000

# ── Chunking defaults ─────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE     = 500
DEFAULT_CHUNK_OVERLAP  = 100
DEFAULT_CHUNK_STRATEGY = "tokens"

# ── Search / retrieval ────────────────────────────────────────────────────────
DEFAULT_TOP_K                = 12
DEFAULT_SIMILARITY_THRESHOLD = 0.25

# ── File validation ───────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB   = 10
ALLOWED_EXTENSIONS = {".pdf"}
MAX_QUERY_LENGTH   = 500

# ── Session management ────────────────────────────────────────────────────────
SESSIONS_DIR          = ROOT_DIR / "data" / "sessions"
SESSION_TIMEOUT_HOURS = 24
AUTO_SAVE_EVERY       = 5

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR  = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# ── Uploads ───────────────────────────────────────────────────────────────────
UPLOADS_DIR = ROOT_DIR / "uploads"

# ── QA Engine ─────────────────────────────────────────────────────────────────
QA_MAX_CONTEXT_TOKENS      = 32_000
QA_TEMPERATURE             = 0.2
CONVERSATION_HISTORY_TURNS = 3

# ── Summarisation ─────────────────────────────────────────────────────────────
SUMMARY_CHUNKS = 5