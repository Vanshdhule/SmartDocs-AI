# backend/search_engine.py
"""
Semantic search + fault-tolerant RAG pipeline — workspace-aware.
Searches within the active workspace's ChromaDB collection only.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

import numpy as np

from .embeddings import EmbeddingGenerator
from .vector_db  import VectorDatabase, VectorDBError
from .qa_engine  import QAEngine

try:
    from config import (
        DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD,
        PRIMARY_MODEL, SECONDARY_MODEL, DEFAULT_WORKSPACE_NAME,
    )
except ImportError:
    DEFAULT_TOP_K                = 12
    DEFAULT_SIMILARITY_THRESHOLD = 0.25
    PRIMARY_MODEL                = "meta/llama-3.3-70b-instruct"
    SECONDARY_MODEL              = "nvidia/llama-3.1-nemotron-70b-instruct"
    DEFAULT_WORKSPACE_NAME       = "Default"

logger = logging.getLogger("smartdocs.search")


class SearchEngine:

    def __init__(
        self,
        model:     str = PRIMARY_MODEL,
        workspace: str = DEFAULT_WORKSPACE_NAME,
    ):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db           = VectorDatabase(workspace=workspace)
        self.qa_engine           = QAEngine(model=model)
        self.active_model        = model
        self.active_workspace    = workspace

    # ── Workspace switching ────────────────────────────────────────────────────

    def switch_workspace(self, workspace: str) -> None:
        """Point the search engine at a different workspace collection."""
        if workspace != self.active_workspace:
            self.vector_db.set_workspace(workspace)
            self.active_workspace = workspace
            self.qa_engine.clear_memory()   # clear context from old workspace
            logger.info(f"SearchEngine workspace switched to '{workspace}'")

    # ── Model switching ────────────────────────────────────────────────────────

    def switch_model(self, new_model: str) -> None:
        if new_model != self.active_model:
            self.qa_engine.switch_model(new_model)
            self.active_model = new_model

    # ── High-level RAG entry ───────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline with three-tier fallback:
        Tier 1: Primary LLM  →  Tier 2: Secondary LLM  →  Tier 3: Raw chunks
        """
        chunks = self.search_similar_chunks(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
            rerank=True,
        )

        if not chunks:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    f"**{self.active_workspace}** workspace. "
                    "Try lowering the similarity threshold or check that "
                    "the correct workspace is selected."
                ),
                "citations":            [],
                "quality_check_passed": False,
                "no_results":           True,
                "fallback_used":        False,
                "fallback_tier":        0,
                "model_used":           self.active_model,
                "workspace":            self.active_workspace,
            }

        # ── Tier 1: Primary model ─────────────────────────────────────────────
        try:
            result = self.qa_engine.generate_answer(
                question=question, retrieved_chunks=chunks
            )
            result["fallback_used"] = False
            result["fallback_tier"] = 1
            result["workspace"]     = self.active_workspace
            return result
        except Exception as tier1_err:
            logger.warning(f"Tier 1 failed ({self.active_model}): {tier1_err}")

        # ── Tier 2: Secondary model ───────────────────────────────────────────
        secondary = SECONDARY_MODEL if SECONDARY_MODEL != self.active_model else PRIMARY_MODEL
        try:
            logger.info(f"Tier 2: trying '{secondary}'")
            self.qa_engine.switch_model(secondary)
            self.active_model = secondary
            result = self.qa_engine.generate_answer(
                question=question, retrieved_chunks=chunks
            )
            result["fallback_used"] = True
            result["fallback_tier"] = 2
            result["fallback_note"] = f"Primary unavailable — answered by {secondary}"
            result["workspace"]     = self.active_workspace
            return result
        except Exception as tier2_err:
            logger.error(f"Tier 2 failed ({secondary}): {tier2_err}")

        # ── Tier 3: Raw chunk fallback ────────────────────────────────────────
        return self._raw_chunk_fallback(
            question, chunks, f"Tier1: {tier1_err} | Tier2: {tier2_err}"
        )

    def _raw_chunk_fallback(
        self, question: str, chunks: List[Dict], error_detail: str
    ) -> Dict[str, Any]:
        lines = [
            "⚠️ *Both AI models temporarily unavailable. "
            "Here are the most relevant passages:*\n"
        ]
        citations = []
        for i, chunk in enumerate(chunks[:3], 1):
            src  = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            lines.append(f"**[{i}] {src} — Page {page}**")
            lines.append(f"{chunk.get('text','')[:400]}…\n")
            citations.append({"file": src, "page": page})
        lines.append(f"\n*Error: {error_detail[:100]}*")
        return {
            "answer":               "\n".join(lines),
            "citations":            citations,
            "quality_check_passed": False,
            "fallback_used":        True,
            "fallback_tier":        3,
            "fallback_note":        "Both models failed — raw chunks shown",
            "model_used":           "none",
            "workspace":            self.active_workspace,
            "reasoning_trace":      None,
        }

    def _fallback_answer(self, q, chunks, err):
        return self._raw_chunk_fallback(q, chunks, err)

    # ── Core vector search ─────────────────────────────────────────────────────

    def search_similar_chunks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:

        logger.info(
            f"Search: {query!r} | workspace='{self.active_workspace}' "
            f"| top_k={top_k} | threshold={similarity_threshold}"
            + (f" | filter={filters}" if filters else "")
        )

        query_embedding = self.embedding_generator.generate_embedding(query)

        raw = self.vector_db.query_documents(
            query_embedding=query_embedding,
            n_results=top_k * 3,
            where=filters,
        )

        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return []

        transformed = [
            {
                "id":        raw["ids"][0][i],
                "text":      raw["documents"][0][i],
                "metadata":  raw["metadatas"][0][i],
                "embedding": raw["embeddings"][0][i] if "embeddings" in raw else None,
                "distance":  raw["distances"][0][i]  if "distances"  in raw else None,
            }
            for i in range(len(raw["ids"][0]))
        ]

        scored   = self.calculate_relevance_score(query_embedding, transformed)
        filtered = self.filter_by_threshold(scored, similarity_threshold)

        if not filtered:
            return []

        if rerank:
            filtered = self.rerank_results(filtered)

        return self.format_search_results(filtered[:top_k])

    def calculate_relevance_score(
        self, query_embedding: List[float], results: List[Dict]
    ) -> List[Dict]:
        q      = np.array(query_embedding)
        scored = []
        for r in results:
            if r.get("embedding") is not None:
                v   = np.array(r["embedding"])
                sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10))
            elif r.get("distance") is not None:
                sim = round(1.0 / (1.0 + float(r["distance"])), 4)
            else:
                sim = 0.0
            r["relevance_score"] = round(sim, 4)
            scored.append(r)
        return scored

    def filter_by_threshold(self, results: List[Dict], threshold: float) -> List[Dict]:
        return [r for r in results if r.get("relevance_score", 0.0) >= threshold]

    def rerank_results(self, results: List[Dict]) -> List[Dict]:
        def _s(r):
            boost      = 0.0
            created_at = r.get("metadata", {}).get("created_at")
            if isinstance(created_at, datetime):
                age_days = (datetime.utcnow() - created_at).days
                boost    = max(0.0, 1.0 - age_days / 365)
            return r["relevance_score"] + 0.05 * boost
        return sorted(results, key=_s, reverse=True)

    def format_search_results(self, results: List[Dict]) -> List[Dict]:
        return [
            {
                "text":            r["text"],
                "source":          r["metadata"].get("source_file"),
                "page":            r["metadata"].get("page_number"),
                "chunk_index":     r["metadata"].get("chunk_index"),
                "relevance_score": r.get("relevance_score"),
                "workspace":       r["metadata"].get("workspace", self.active_workspace),
            }
            for r in results
        ]

    def get_source_distribution(self, retrieved_chunks: List[Dict]) -> Dict:
        counts: Dict[str, int] = {}
        for chunk in retrieved_chunks:
            src = chunk.get("source") or chunk.get("metadata", {}).get("source_file") or "Unknown"
            counts[src] = counts.get(src, 0) + 1
        total       = sum(counts.values()) or 1
        percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
        return {"counts": counts, "percentages": percentages, "total_chunks": total}