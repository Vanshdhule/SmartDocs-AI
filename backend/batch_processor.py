# backend/batch_processor.py
"""Parallel PDF ingestion — workspace-aware."""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional

from .ingestion_pipeline import DocumentIngestion

logger      = logging.getLogger("smartdocs.batch")
MAX_WORKERS = 3

try:
    from config import DEFAULT_WORKSPACE_NAME
except ImportError:
    DEFAULT_WORKSPACE_NAME = "Default"


class BatchProcessor:

    def __init__(self):
        self._pipeline_factory = DocumentIngestion

    def process_files(
        self,
        files:             List,
        workspace:         str  = DEFAULT_WORKSPACE_NAME,
        clean_text:        bool = True,
        chunk_strategy:    str  = "tokens",
        chunk_size:        int  = 500,
        chunk_overlap:     int  = 100,
        replace_existing:  bool = False,
        generate_summary:  bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:

        total        = len(files)
        results      = []
        done         = 0
        total_chunks = 0
        total_pages  = 0
        start        = time.time()

        logger.info(f"BatchProcessor: {total} file(s), workspace='{workspace}'")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    self._process_one, f, workspace,
                    clean_text, chunk_strategy, chunk_size,
                    chunk_overlap, replace_existing, generate_summary,
                ): f
                for f in files
            }
            for future in as_completed(future_map):
                uploaded = future_map[future]
                fname    = getattr(uploaded, "name", str(uploaded))
                done    += 1
                try:
                    result = future.result()
                    results.append(result)
                    if result.get("success") and not result.get("already_indexed"):
                        total_chunks += result.get("chunks", 0)
                        total_pages  += result.get("pages",  0)
                    logger.info(f"[{done}/{total}] {fname} done")
                except Exception as e:
                    logger.error(f"[{done}/{total}] {fname} FAILED: {e}")
                    results.append({
                        "file_name": fname, "success": False, "error": str(e)
                    })
                if progress_callback:
                    try: progress_callback(done, total, fname)
                    except Exception: pass

        elapsed      = round(time.time() - start, 2)
        chunks_per_s = round(total_chunks / max(elapsed, 0.001), 1)
        logger.info(
            f"BatchProcessor: done in {elapsed}s — "
            f"{total_chunks} chunks ({chunks_per_s}/s), workspace='{workspace}'"
        )
        return {
            "results":         results,
            "successful":      sum(1 for r in results if r.get("success")),
            "failed":          sum(1 for r in results if not r.get("success")),
            "total_chunks":    total_chunks,
            "total_pages":     total_pages,
            "elapsed_seconds": elapsed,
            "chunks_per_sec":  chunks_per_s,
            "workspace":       workspace,
        }

    def _process_one(
        self,
        uploaded_file,
        workspace:        str,
        clean_text:       bool,
        chunk_strategy:   str,
        chunk_size:       int,
        chunk_overlap:    int,
        replace_existing: bool,
        generate_summary: bool,
    ) -> Dict[str, Any]:
        pipeline = self._pipeline_factory(workspace=workspace)
        fname    = getattr(uploaded_file, "name", "unknown.pdf")
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return pipeline.process_single_document(
                uploaded_file,
                clean_text=clean_text,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                replace_existing=replace_existing,
                generate_summary=generate_summary,
            )
        except Exception as e:
            logger.error(f"_process_one failed for {fname}: {e}")
            return {"file_name": fname, "success": False, "error": str(e)}

    @staticmethod
    def stage_progress(stage: str) -> float:
        return {
            "extract": 0.20, "clean": 0.35, "chunk": 0.40,
            "embed": 0.65, "store": 0.80, "summarise": 1.00,
        }.get(stage, 0.0)