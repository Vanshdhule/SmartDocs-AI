# utils/error_handler.py
"""
Centralised error handling, input validation, and system health check.
All logging goes through utils/logging_config — no basicConfig calls here.
"""

import os
import logging
from pathlib import Path

try:
    from config import MAX_FILE_SIZE_MB, MAX_QUERY_LENGTH, ALLOWED_EXTENSIONS
except ImportError:
    MAX_FILE_SIZE_MB   = 10
    MAX_QUERY_LENGTH   = 500
    ALLOWED_EXTENSIONS = {".pdf"}

logger = logging.getLogger("smartdocs.error_handler")


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SmartDocsBaseError(Exception):
    def __init__(self, message: str, user_message: str = None):
        super().__init__(message)
        self.user_message = user_message or message

class PDFProcessingError(SmartDocsBaseError): pass
class EmbeddingError(SmartDocsBaseError):     pass
class DatabaseError(SmartDocsBaseError):      pass
class APIError(SmartDocsBaseError):           pass
class ValidationError(SmartDocsBaseError):    pass


# ── Input validation ──────────────────────────────────────────────────────────

def validate_uploaded_file(file) -> None:
    if file is None:
        raise ValidationError("No file provided.", "Please upload a PDF file.")
    name = getattr(file, "name", "")
    ext  = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Invalid file type: {ext}",
            f"Only PDF files are accepted. Got: {ext or 'unknown'}",
        )
    size_mb = getattr(file, "size", 0) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(
            f"File too large: {size_mb:.1f} MB",
            f"Max file size is {MAX_FILE_SIZE_MB} MB. "
            f"'{name}' is {size_mb:.1f} MB.",
        )


def validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValidationError(
            "Empty query.", "Please enter a question before searching."
        )
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query too long: {len(query)} chars",
            f"Question must be {MAX_QUERY_LENGTH} characters or fewer.",
        )
    for dangerous in ["<script", "javascript:", "\x00"]:
        if dangerous.lower() in query.lower():
            raise ValidationError(
                "Unsafe query content.",
                "Your question contains unsupported characters. Please rephrase.",
            )
    return query


# ── OpenAI error classifier ───────────────────────────────────────────────────

def classify_api_error(error: Exception) -> str:
    """Classify NVIDIA NIM / OpenAI-compatible API errors into user-friendly messages."""
    msg = str(error).lower()
    if "api key" in msg or "authentication" in msg or "unauthorized" in msg:
        return "❌ Invalid or missing API key. Check NVIDIA_CHAT_API_KEY in your .env file."
    if "rate limit" in msg:
        return "⚠️ Rate limit reached. Please wait a moment and try again."
    if "quota" in msg or "billing" in msg or "insufficient" in msg:
        return "💳 API quota exceeded. Check your NVIDIA NIM usage at build.nvidia.com."
    if "timeout" in msg or "connection" in msg or "network" in msg:
        return "🌐 Network error. Check your internet connection."
    if "model" in msg and "not found" in msg:
        return "🤖 The requested model is unavailable. Check your NVIDIA NIM account."
    return f"⚠️ API error: {str(error)[:120]}"


# ── System health check ───────────────────────────────────────────────────────

def check_system_health() -> dict:
    """Check ChromaDB connectivity, API key, and available disk space."""
    results = {}

    # 1. Database
    try:
        from backend.vector_db import VectorDatabase
        db = VectorDatabase()
        ok = db.verify_connection()
        results["database"] = {
            "ok":      ok,
            "message": "ChromaDB healthy" if ok else "ChromaDB not reachable",
        }
    except Exception as e:
        results["database"] = {"ok": False, "message": f"DB error: {str(e)[:80]}"}

    # 2. NVIDIA API keys
    try:
        from dotenv import load_dotenv
        load_dotenv()
        chat_key  = os.getenv("NVIDIA_CHAT_API_KEY", "")
        emb_key   = os.getenv("NVIDIA_EMBEDDING_API_KEY", "")
        key_ok    = (
            bool(chat_key) and chat_key != "nvapi-your-chat-key-here"
            and bool(emb_key) and emb_key != "nvapi-your-embedding-key-here"
        )
        results["api_key"] = {
            "ok":      key_ok,
            "message": "NVIDIA API keys loaded" if key_ok
                       else "NVIDIA_CHAT_API_KEY or NVIDIA_EMBEDDING_API_KEY missing",
        }
    except Exception as e:
        results["api_key"] = {"ok": False, "message": str(e)[:80]}

    # 3. Disk space (warn below 500 MB)
    try:
        import shutil
        free_mb = shutil.disk_usage(".").free / (1024 * 1024)
        disk_ok = free_mb > 500
        results["disk"] = {
            "ok":      disk_ok,
            "message": f"{free_mb:.0f} MB free"
            if disk_ok else f"⚠️ Low disk space: {free_mb:.0f} MB",
        }
    except Exception:
        results["disk"] = {"ok": True, "message": "Disk check skipped"}

    results["overall"] = all(v["ok"] for v in results.values())
    logger.info(f"Health check: {results}")
    return results

# Backwards-compatibility alias
classify_openai_error = classify_api_error