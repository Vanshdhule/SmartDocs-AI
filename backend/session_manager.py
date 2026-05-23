# backend/session_manager.py
"""
Session Management — now workspace-aware.
Each session tracks its active workspace and workspace history.
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from config import (
        SESSIONS_DIR, SESSION_TIMEOUT_HOURS, AUTO_SAVE_EVERY,
        DEFAULT_WORKSPACE_NAME,
    )
except ImportError:
    SESSIONS_DIR          = Path("data/sessions")
    SESSION_TIMEOUT_HOURS = 24
    AUTO_SAVE_EVERY       = 5
    DEFAULT_WORKSPACE_NAME = "Default"

logger = logging.getLogger("smartdocs.session")


class SessionManager:

    def __init__(self):
        Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
        self.cleanup_expired_sessions()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create_session(self) -> Dict[str, Any]:
        session = {
            "session_id":       str(uuid.uuid4()),
            "created_at":       datetime.utcnow().isoformat(),
            "last_active":      datetime.utcnow().isoformat(),
            "chat_history":     [],
            "uploaded_docs":    [],
            "total_chunks":     0,
            "total_pages":      0,
            "query_count":      0,
            "bookmarks":        [],
            "response_times":   [],
            # ── Workspace state ────────────────────────────────────────────────
            "active_workspace": DEFAULT_WORKSPACE_NAME,
            "workspaces": {
                DEFAULT_WORKSPACE_NAME: {
                    "name":         DEFAULT_WORKSPACE_NAME,
                    "created_at":   datetime.utcnow().isoformat(),
                    "uploaded_docs": [],
                    "total_chunks":  0,
                    "total_pages":   0,
                }
            },
            "preferences": {
                "dark_mode":    False,
                "chunk_method": "tokens",
                "chunk_size":   500,
                "overlap":      100,
            },
        }
        logger.info(f"Session created: {session['session_id']}")
        return session

    # ── Workspace helpers ─────────────────────────────────────────────────────

    def get_active_workspace(self, session: Dict[str, Any]) -> str:
        return session.get("active_workspace", DEFAULT_WORKSPACE_NAME)

    def set_active_workspace(
        self, session: Dict[str, Any], workspace_name: str
    ) -> None:
        session["active_workspace"] = workspace_name
        # Ensure workspace entry exists
        if workspace_name not in session.get("workspaces", {}):
            session.setdefault("workspaces", {})[workspace_name] = {
                "name":          workspace_name,
                "created_at":    datetime.utcnow().isoformat(),
                "uploaded_docs": [],
                "total_chunks":  0,
                "total_pages":   0,
            }

    def create_workspace(
        self, session: Dict[str, Any], workspace_name: str
    ) -> bool:
        """Create a new workspace. Returns False if it already exists."""
        workspaces = session.setdefault("workspaces", {})
        if workspace_name in workspaces:
            return False
        workspaces[workspace_name] = {
            "name":          workspace_name,
            "created_at":    datetime.utcnow().isoformat(),
            "uploaded_docs": [],
            "total_chunks":  0,
            "total_pages":   0,
        }
        logger.info(f"Workspace created: '{workspace_name}'")
        return True

    def get_workspace_docs(
        self, session: Dict[str, Any], workspace_name: str = None
    ) -> List[Dict]:
        """Get uploaded_docs for a specific workspace (or active one)."""
        ws = workspace_name or self.get_active_workspace(session)
        workspaces = session.get("workspaces", {})
        return workspaces.get(ws, {}).get("uploaded_docs", [])

    def add_doc_to_workspace(
        self,
        session:        Dict[str, Any],
        workspace_name: str,
        doc_entry:      Dict[str, Any],
    ) -> None:
        """Add an indexed document entry to a workspace."""
        workspaces = session.setdefault("workspaces", {})
        ws_data    = workspaces.setdefault(workspace_name, {
            "name": workspace_name, "created_at": datetime.utcnow().isoformat(),
            "uploaded_docs": [], "total_chunks": 0, "total_pages": 0,
        })
        existing_names = [d["name"] for d in ws_data.get("uploaded_docs", [])]
        if doc_entry["name"] not in existing_names:
            ws_data.setdefault("uploaded_docs", []).append(doc_entry)
            ws_data["total_chunks"] = ws_data.get("total_chunks", 0) + doc_entry.get("chunks", 0)
            ws_data["total_pages"]  = ws_data.get("total_pages",  0) + doc_entry.get("pages",  0)

    def remove_doc_from_workspace(
        self, session: Dict[str, Any], workspace_name: str, doc_name: str
    ) -> None:
        ws_data = session.get("workspaces", {}).get(workspace_name, {})
        ws_data["uploaded_docs"] = [
            d for d in ws_data.get("uploaded_docs", [])
            if d["name"] != doc_name
        ]

    def list_workspace_names(self, session: Dict[str, Any]) -> List[str]:
        return list(session.get("workspaces", {DEFAULT_WORKSPACE_NAME: {}}).keys())

    # ── Persist / load ────────────────────────────────────────────────────────

    def save_session_state(self, session: Dict[str, Any]) -> str:
        session["last_active"] = datetime.utcnow().isoformat()
        sid  = session.get("session_id", "unknown")
        path = Path(SESSIONS_DIR) / f"{sid}.json"
        serialisable = json.loads(json.dumps(session, default=str))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        logger.info(f"Session saved: {path}")
        return str(path)

    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = Path(SESSIONS_DIR) / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)
        last_active = datetime.fromisoformat(
            session.get("last_active", "2000-01-01")
        )
        if datetime.utcnow() - last_active > timedelta(hours=SESSION_TIMEOUT_HOURS):
            path.unlink(missing_ok=True)
            return None
        return session

    def list_sessions(self) -> List[Dict[str, str]]:
        sessions = []
        for p in Path(SESSIONS_DIR).glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id":  data.get("session_id", p.stem),
                    "created_at":  data.get("created_at", ""),
                    "query_count": str(data.get("query_count", 0)),
                })
            except Exception:
                pass
        return sessions

    def cleanup_expired_sessions(self) -> int:
        removed = 0
        for p in Path(SESSIONS_DIR).glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                last_active = datetime.fromisoformat(
                    data.get("last_active", "2000-01-01")
                )
                if datetime.utcnow() - last_active > timedelta(
                    hours=SESSION_TIMEOUT_HOURS
                ):
                    p.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                p.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} expired session(s)")
        return removed

    # ── Auto-save ─────────────────────────────────────────────────────────────

    def should_auto_save(self, query_count: int) -> bool:
        return query_count > 0 and query_count % AUTO_SAVE_EVERY == 0

    # ── Bookmarks ─────────────────────────────────────────────────────────────

    def add_bookmark(self, session: Dict[str, Any], index: int) -> None:
        if index not in session.setdefault("bookmarks", []):
            session["bookmarks"].append(index)

    def remove_bookmark(self, session: Dict[str, Any], index: int) -> None:
        session.setdefault("bookmarks", [])
        if index in session["bookmarks"]:
            session["bookmarks"].remove(index)

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(self, session: Dict[str, Any]) -> Dict[str, Any]:
        times = session.get("response_times", [])
        ws    = self.get_active_workspace(session)
        ws_data = session.get("workspaces", {}).get(ws, {})
        return {
            "session_id":       session.get("session_id", "—"),
            "created_at":       session.get("created_at", "—")[:19].replace("T"," "),
            "query_count":      session.get("query_count", 0),
            "active_workspace": ws,
            "workspace_count":  len(session.get("workspaces", {})),
            "documents":        len(ws_data.get("uploaded_docs", [])),
            "total_chunks":     ws_data.get("total_chunks", 0),
            "total_pages":      ws_data.get("total_pages", 0),
            "bookmarks":        len(session.get("bookmarks", [])),
            "avg_response_s":   round(sum(times)/len(times), 2) if times else 0,
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def _ts(self, m):
        ts = m.get("timestamp", "")
        if isinstance(ts, datetime): return ts.strftime("%H:%M")
        if isinstance(ts, str) and "T" in ts: return ts[11:16]
        return str(ts)[:5]

    def export_as_text(self, session: Dict[str, Any]) -> str:
        lines = [
            "SmartDocs AI – Conversation Export",
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Session ID: {session.get('session_id','')}",
            f"Workspace:  {self.get_active_workspace(session)}",
            "=" * 60, "",
        ]
        for i, msg in enumerate(session.get("chat_history", [])):
            role = "You" if msg["role"] == "user" else "SmartDoc AI"
            lines.append(f"[{self._ts(msg)}] {role}:")
            lines.append(msg.get("content", ""))
            if i in session.get("bookmarks", []):
                lines.append("⭐ Bookmarked")
            lines.append("")
        return "\n".join(lines)

    def export_as_markdown(self, session: Dict[str, Any]) -> str:
        stats = self.get_statistics(session)
        lines = [
            "# SmartDocs AI – Conversation Export", "",
            f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Session:** `{stats['session_id']}`  ",
            f"**Workspace:** {stats['active_workspace']}  ",
            f"**Queries:** {stats['query_count']} | "
            f"**Documents:** {stats['documents']}",
            "", "---", "",
        ]
        for i, msg in enumerate(session.get("chat_history", [])):
            role     = "**You**" if msg["role"] == "user" else "**SmartDoc AI**"
            bookmark = " ⭐" if i in session.get("bookmarks", []) else ""
            lines.append(f"### {role}{bookmark} _{self._ts(msg)}_")
            lines.append("")
            lines.append(msg.get("content", ""))
            lines.append("")
        return "\n".join(lines)