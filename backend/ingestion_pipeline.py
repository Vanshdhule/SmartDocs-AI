# backend/ingestion_pipeline.py
"""
End-to-end ingestion pipeline — workspace-aware.
Each pipeline instance targets a specific workspace (ChromaDB collection).
"""

import uuid
import time
import logging
import os
from typing import List, Dict, Any

from .pdf_processor  import PDFProcessor, PDFProcessingError
from .text_cleaner   import TextCleaner
from .text_chunker   import TextChunker
from .embeddings     import EmbeddingGenerator
from .vector_db      import VectorDatabase, VectorDBError

try:
    from config import (
        DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_STRATEGY,
        SUMMARY_CHUNKS, DEFAULT_WORKSPACE_NAME,
    )
except ImportError:
    DEFAULT_CHUNK_SIZE      = 500
    DEFAULT_CHUNK_OVERLAP   = 100
    DEFAULT_CHUNK_STRATEGY  = "tokens"
    SUMMARY_CHUNKS          = 5
    DEFAULT_WORKSPACE_NAME  = "Default"

logger = logging.getLogger("smartdocs.ingestion")


class DocumentIngestion:

    STAGES = [
        "INITIALIZED", "PDF_EXTRACTION", "TEXT_CLEANING",
        "CHUNKING", "EMBEDDING", "VECTOR_STORAGE",
        "SUMMARISING", "COMPLETED", "FAILED",
    ]

    def __init__(self, workspace: str = DEFAULT_WORKSPACE_NAME):
        self.pdf_processor       = PDFProcessor()
        self.text_cleaner        = TextCleaner()
        self.text_chunker        = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db           = VectorDatabase(workspace=workspace)
        self.workspace           = workspace

        self.current_stage   = "INITIALIZED"
        self.files_processed = 0
        self.total_files     = 0
        self.start_time      = None

    def switch_workspace(self, workspace: str) -> None:
        """Switch this pipeline instance to a different workspace."""
        self.workspace = workspace
        self.vector_db.set_workspace(workspace)
        logger.info(f"Pipeline workspace switched to '{workspace}'")

    def get_processing_status(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "stage":            self.current_stage,
            "workspace":        self.workspace,
            "files_processed":  self.files_processed,
            "total_files":      self.total_files,
            "progress_percent": int(
                (self.files_processed / self.total_files) * 100
            ) if self.total_files else 0,
            "elapsed_seconds":  round(elapsed, 2),
        }

    def rollback_on_error(self, document_ids: List[str]) -> None:
        try:
            if document_ids:
                logger.warning(f"Rolling back {len(document_ids)} partial inserts")
                self.vector_db.delete_documents(document_ids)
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def process_single_document(
        self,
        uploaded_file,
        clean_text:       bool = True,
        chunk_strategy:   str  = DEFAULT_CHUNK_STRATEGY,
        chunk_size:       int  = DEFAULT_CHUNK_SIZE,
        chunk_overlap:    int  = DEFAULT_CHUNK_OVERLAP,
        replace_existing: bool = False,
        generate_summary: bool = True,
    ) -> Dict[str, Any]:

        start_time         = time.time()
        fname              = getattr(uploaded_file, "name", "unknown.pdf")
        document_chunk_ids = []
        logger.info(f"[{self.workspace}] Starting ingestion for '{fname}'")

        try:
            # ── Deduplication ────────────────────────────────────────────────
            existing_ids = self.vector_db.get_source_file_ids(fname)
            if existing_ids:
                if replace_existing:
                    logger.info(f"Replacing {len(existing_ids)} chunks for '{fname}'")
                    self.vector_db.delete_documents(existing_ids)
                else:
                    logger.info(f"Skipping '{fname}' — already indexed in '{self.workspace}'")
                    return {
                        "file_name":       fname,
                        "success":         True,
                        "already_indexed": True,
                        "existing_chunks": len(existing_ids),
                        "pages":           0,
                        "chunks":          len(existing_ids),
                        "processing_time": 0.0,
                        "chunks_per_sec":  0.0,
                        "summary":         "",
                        "workspace":       self.workspace,
                    }

            # ── 1. Extract ───────────────────────────────────────────────────
            self.current_stage = "PDF_EXTRACTION"
            raw_pages = self.pdf_processor.extract_text(uploaded_file)
            if not raw_pages:
                raise ValueError("No text extracted from PDF")

            pages = []
            for i, page in enumerate(raw_pages):
                if isinstance(page, dict):
                    pages.append(page)
                elif isinstance(page, str):
                    pages.append({"page_number": i + 1, "text": page})

            # ── 2. Clean ─────────────────────────────────────────────────────
            self.current_stage = "TEXT_CLEANING"
            cleaned_pages = []
            for page in pages:
                text = (
                    self.text_cleaner.clean_text(page["text"])
                    if clean_text else page["text"]
                )
                if text.strip():
                    cleaned_pages.append({"page_number": page["page_number"], "text": text})

            if not cleaned_pages:
                raise ValueError("All pages empty after cleaning")

            # ── 3. Chunk ─────────────────────────────────────────────────────
            self.current_stage = "CHUNKING"
            self.text_chunker.default_chunk_size = chunk_size
            self.text_chunker.default_overlap    = chunk_overlap

            all_chunks: List[Dict[str, Any]] = []
            for page in cleaned_pages:
                for chunk in self.text_chunker.create_chunks(
                    text=page["text"],
                    source_file=fname,
                    page_number=page["page_number"],
                    strategy=chunk_strategy,
                ):
                    all_chunks.append(chunk)

            if not all_chunks:
                raise ValueError("Chunking produced no output")

            # ── 4. Embed + Store ─────────────────────────────────────────────
            self.current_stage = "EMBEDDING"
            t_embed = time.time()
            try:
                embedded = self.embedding_generator.prepare_embedding_data(all_chunks)
                for chunk in embedded:
                    chunk["chunk_id"] = f"{fname}_{self.workspace}_{uuid.uuid4().hex}"
                self.current_stage = "VECTOR_STORAGE"
                self.vector_db.add_documents(embedded)
            except Exception as e:
                raise RuntimeError(f"Embedding/storage failed: {e}")

            document_chunk_ids = [c["chunk_id"] for c in embedded]
            embed_time         = max(time.time() - t_embed, 0.001)
            chunks_per_sec     = round(len(embedded) / embed_time, 1)

            # ── 5. Summarise ─────────────────────────────────────────────────
            summary = ""
            if generate_summary:
                self.current_stage = "SUMMARISING"
                try:
                    from .qa_engine import QAEngine
                    qa             = QAEngine()
                    summary_chunks = [
                        {"page_number": c["page_number"], "text": c["text"]}
                        for c in all_chunks[:SUMMARY_CHUNKS]
                    ]
                    summary = qa.summarize_document(summary_chunks, fname)
                except Exception as e:
                    logger.warning(f"Summarisation skipped for '{fname}': {e}")

            processing_time = round(time.time() - start_time, 2)
            logger.info(
                f"[{self.workspace}] Ingestion complete: '{fname}' — "
                f"{len(pages)} pages, {len(embedded)} chunks, {chunks_per_sec}/s"
            )
            return {
                "file_name":       fname,
                "pages":           len(pages),
                "chunks":          len(embedded),
                "processing_time": processing_time,
                "chunks_per_sec":  chunks_per_sec,
                "success":         True,
                "already_indexed": False,
                "summary":         summary,
                "workspace":       self.workspace,
            }

        except Exception as e:
            self.current_stage = "FAILED"
            logger.error(f"[{self.workspace}] Ingestion failed for '{fname}': {e}")
            self.rollback_on_error(document_chunk_ids)
            return {"file_name": fname, "success": False,
                    "error": str(e), "workspace": self.workspace}

    def process_multiple_documents(
        self,
        pdf_paths: List = None,
        pdf_files: List = None,
        uploaded_files: List = None,
    ) -> Dict[str, Any]:
        files = pdf_paths or pdf_files or uploaded_files
        if not files:
            raise ValueError("No PDF files provided")
        self.start_time      = time.time()
        self.total_files     = len(files)
        self.files_processed = 0
        results, errors      = [], []
        for item in files:
            try:
                if isinstance(item, (str, bytes, os.PathLike)):
                    with open(item, "rb") as f:
                        result = self.process_single_document(f)
                    file_name = os.path.basename(item)
                else:
                    result    = self.process_single_document(item)
                    file_name = getattr(item, "name", "unknown.pdf")
                results.append(result)
            except Exception as e:
                fname = getattr(item, "name", str(item))
                errors.append({"file": fname, "error": str(e)})
                results.append({"file_name": fname, "success": False, "error": str(e)})
            self.files_processed += 1
        self.current_stage = "COMPLETED"
        return {
            "results":    results,
            "successful": sum(1 for r in results if r.get("success")),
            "failed":     sum(1 for r in results if not r.get("success")),
            "errors":     errors,
        }