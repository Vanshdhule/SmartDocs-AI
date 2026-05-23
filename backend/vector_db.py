# backend/vector_db.py
"""
ChromaDB vector store wrapper — Workspace-aware.

Each workspace is a separate ChromaDB collection:
    Collection name = "smartdocs_<workspace_slug>"

This gives complete isolation between unrelated document sets
while still supporting combined search across all docs in a workspace.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional

import chromadb

try:
    from config import (
        CHROMA_PERSIST_DIR, WORKSPACE_COLLECTION_PREFIX,
        DEFAULT_WORKSPACE_NAME, EMBEDDING_DIMENSION,
    )
except ImportError:
    CHROMA_PERSIST_DIR           = "data/chroma_db"
    WORKSPACE_COLLECTION_PREFIX  = "smartdocs_"
    DEFAULT_WORKSPACE_NAME       = "Default"
    EMBEDDING_DIMENSION          = 1024

logger = logging.getLogger("smartdocs.vector_db")


class VectorDBError(Exception):
    pass


def workspace_to_collection_name(workspace_name: str) -> str:
    """
    Convert a human-readable workspace name to a valid ChromaDB collection name.
    Rules: lowercase, alphanumeric + underscores only, max 63 chars.
    e.g. "AI Research 2025" → "smartdocs_ai_research_2025"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", workspace_name.lower().strip())
    slug = slug.strip("_")[:50]  # leave room for prefix
    return f"{WORKSPACE_COLLECTION_PREFIX}{slug}"


class VectorDatabase:
    """
    Workspace-aware ChromaDB wrapper.
    Instantiate with a workspace name to get an isolated collection.
    """

    def __init__(
        self,
        persist_dir: str  = CHROMA_PERSIST_DIR,
        workspace:   str  = DEFAULT_WORKSPACE_NAME,
    ):
        try:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            self.persist_directory = persist_dir
            self.client            = chromadb.PersistentClient(path=persist_dir)
            self.set_workspace(workspace)
        except Exception as e:
            raise VectorDBError(f"Failed to initialise vector database: {e}")

    # ── Workspace management ──────────────────────────────────────────────────

    def set_workspace(self, workspace_name: str) -> None:
        """Switch to a different workspace (collection)."""
        self.workspace_name    = workspace_name
        self.collection_name   = workspace_to_collection_name(workspace_name)
        self.collection        = self._get_or_create_collection()
        logger.info(
            f"VectorDB workspace: '{workspace_name}' "
            f"→ collection: '{self.collection_name}'"
        )

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """
        Return all existing workspaces with their document counts.
        """
        workspaces = []
        for col in self.client.list_collections():
            if col.name.startswith(WORKSPACE_COLLECTION_PREFIX):
                human_name = col.name[len(WORKSPACE_COLLECTION_PREFIX):].replace("_", " ").title()
                try:
                    count = self.client.get_collection(col.name).count()
                except Exception:
                    count = 0
                workspaces.append({
                    "name":            human_name,
                    "collection_name": col.name,
                    "chunk_count":     count,
                })
        return workspaces

    def delete_workspace(self, workspace_name: str) -> None:
        """Delete an entire workspace and all its chunks."""
        col_name = workspace_to_collection_name(workspace_name)
        try:
            self.client.delete_collection(col_name)
            logger.info(f"Workspace deleted: '{workspace_name}'")
        except Exception as e:
            raise VectorDBError(f"Failed to delete workspace '{workspace_name}': {e}")

    def get_workspace_docs(self, workspace_name: str = None) -> List[str]:
        """
        Return distinct source_file names in the given workspace (or current).
        """
        try:
            if workspace_name:
                col = self.client.get_collection(
                    workspace_to_collection_name(workspace_name)
                )
            else:
                col = self.collection
            result = col.get(include=["metadatas"])
            seen, docs = set(), []
            for meta in result.get("metadatas", []):
                src = meta.get("source_file")
                if src and src not in seen:
                    seen.add(src)
                    docs.append(src)
            return docs
        except Exception:
            return []

    def get_workspace_doc_stats(self) -> List[Dict[str, Any]]:
        """
        Return per-document stats (name, chunk_count, page_count) for the
        current workspace. Used by the diagnostic panel in the UI.
        """
        try:
            result  = self.collection.get(include=["metadatas"])
            counts: Dict[str, Dict] = {}
            for meta in result.get("metadatas", []):
                src  = meta.get("source_file", "unknown")
                page = meta.get("page_number", 0)
                if src not in counts:
                    counts[src] = {"chunks": 0, "max_page": 0}
                counts[src]["chunks"] += 1
                if page > counts[src]["max_page"]:
                    counts[src]["max_page"] = page
            return [
                {"name": src, "chunks": v["chunks"], "pages": v["max_page"]}
                for src, v in sorted(counts.items())
            ]
        except Exception:
            return []

    # ── Collection helpers ────────────────────────────────────────────────────

    def _get_or_create_collection(self):
        existing = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing:
            return self.client.get_collection(self.collection_name)
        return self.client.create_collection(
            name=self.collection_name,
            metadata={"workspace": self.workspace_name,
                      "description": f"SmartDocs AI — {self.workspace_name}"},
        )

    def check_collection_exists(self) -> bool:
        return self.collection_name in [
            c.name for c in self.client.list_collections()
        ]

    def verify_connection(self) -> bool:
        try:
            self.client.list_collections()
            return True
        except Exception:
            return False

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        try:
            ids, embeddings, documents, metadatas = [], [], [], []
            for chunk in chunks:
                ids.append(chunk.get("chunk_id") or str(uuid.uuid4()))
                embeddings.append(chunk["embedding"])
                documents.append(chunk["text"])
                metadatas.append({
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "token_count": chunk["token_count"],
                    "workspace":   self.workspace_name,
                })
            self.collection.add(
                ids=ids, embeddings=embeddings,
                documents=documents, metadatas=metadatas,
            )
        except Exception as e:
            raise VectorDBError(f"Failed to add documents: {e}")

    def query_documents(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            raise VectorDBError(f"Failed to query documents: {e}")

    def delete_documents(self, doc_ids: List[str]) -> None:
        try:
            self.collection.delete(ids=doc_ids)
        except Exception as e:
            raise VectorDBError(f"Failed to delete documents: {e}")

    def get_source_file_ids(self, source_file: str) -> List[str]:
        try:
            result = self.collection.get(
                where={"source_file": source_file}, include=[],
            )
            return result.get("ids", [])
        except Exception as e:
            raise VectorDBError(
                f"Failed to look up chunks for '{source_file}': {e}"
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "collection_name":  self.collection_name,
                "workspace":        self.workspace_name,
                "count":            count,
                "persist_directory": self.persist_directory,
                "is_empty":         count == 0,
            }
        except Exception as e:
            raise VectorDBError(f"Failed to fetch stats: {e}")

    def get_all_documents(self, limit: int = 10) -> Dict[str, Any]:
        try:
            return self.collection.get(limit=limit)
        except Exception as e:
            raise VectorDBError(f"Failed to retrieve documents: {e}")

    def update_document(
        self,
        doc_id:    str,
        embedding: Optional[List[float]] = None,
        document:  Optional[str]         = None,
        metadata:  Optional[Dict]        = None,
    ) -> None:
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding] if embedding else None,
                documents =[document]  if document  else None,
                metadatas =[metadata]  if metadata  else None,
            )
        except Exception as e:
            raise VectorDBError(f"Failed to update document: {e}")