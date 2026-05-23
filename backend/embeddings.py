# backend/embeddings.py
"""
Embedding Generator
Two-level cache: in-memory dict + JSON file on disk.
Embeddings survive app restarts — no redundant API calls.
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

try:
    from config import (
        NVIDIA_BASE_URL, NVIDIA_EMBEDDING_API_KEY,
        EMBEDDING_MODEL, EMBEDDING_DIMENSION,
        MAX_RETRIES, MAX_REQUESTS_PER_MINUTE,
        EMBEDDING_CACHE_FILE, MAX_CACHE_ENTRIES,
    )
except ImportError:
    NVIDIA_BASE_URL          = "https://integrate.api.nvidia.com/v1"
    NVIDIA_EMBEDDING_API_KEY = None
    EMBEDDING_MODEL          = "nvidia/nv-embedqa-e5-v5"
    EMBEDDING_DIMENSION      = 1024
    MAX_RETRIES              = 3
    MAX_REQUESTS_PER_MINUTE  = 60
    EMBEDDING_CACHE_FILE     = Path("data/embedding_cache.json")
    MAX_CACHE_ENTRIES        = 10_000

logger = logging.getLogger("smartdocs.embeddings")


class EmbeddingGenerator:

    def __init__(self):
        load_dotenv()
        api_key = NVIDIA_EMBEDDING_API_KEY or os.getenv("NVIDIA_EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ NVIDIA_EMBEDDING_API_KEY not found. Set it in .env."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=NVIDIA_BASE_URL,
        )
        self._mem_cache: Dict[str, List[float]] = {}
        self._load_disk_cache()

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_disk_cache(self) -> None:
        try:
            path = Path(EMBEDDING_CACHE_FILE)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._mem_cache = {
                    k: v for k, v in data.items()
                    if isinstance(v, list) and len(v) == EMBEDDING_DIMENSION
                }
                logger.info(
                    f"Embedding cache loaded: {len(self._mem_cache)} entries"
                )
        except Exception as e:
            logger.warning(f"Could not load embedding cache ({e}) — starting fresh")
            self._mem_cache = {}

    def _save_disk_cache(self) -> None:
        try:
            path = Path(EMBEDDING_CACHE_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            cache = self._mem_cache
            if len(cache) > MAX_CACHE_ENTRIES:
                keys  = list(cache.keys())
                keep  = keys[len(keys) // 2:]
                cache = {k: cache[k] for k in keep}
                self._mem_cache = cache
                logger.info(f"Cache eviction: keeping {len(keep)} entries")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def cache_size(self) -> int:
        return len(self._mem_cache)

    def clear_cache(self) -> None:
        self._mem_cache = {}
        try:
            p = Path(EMBEDDING_CACHE_FILE)
            if p.exists():
                p.unlink()
        except Exception as e:
            logger.warning(f"Could not delete cache file: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        h = self._hash_text(text)
        if h in self._mem_cache:
            return self._mem_cache[h]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text,
                    extra_body={"input_type": "passage", "truncate": "END"},
                )
                embedding = resp.data[0].embedding
                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Unexpected dimension: {len(embedding)}"
                    )
                self._mem_cache[h] = embedding
                self._save_disk_cache()
                return embedding
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise Exception(
                        f"Embedding failed after {MAX_RETRIES} attempts: {e}"
                    )
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt} failed, retrying in {wait}s: {e}")
                time.sleep(wait)

    def generate_batch_embeddings(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results    = []
        api_calls  = 0
        cache_hits = 0

        for chunk in chunks:
            was_cached = self._hash_text(chunk["text"]) in self._mem_cache
            embedding  = self.generate_embedding(chunk["text"])

            if was_cached:
                cache_hits += 1
            else:
                api_calls += 1
                if api_calls % MAX_REQUESTS_PER_MINUTE == 0:
                    time.sleep(1)

            enriched              = dict(chunk)
            enriched["embedding"] = embedding
            enriched["timestamp"] = datetime.utcnow().isoformat()
            results.append(enriched)

        logger.info(
            f"Batch embeddings done — {api_calls} API calls, "
            f"{cache_hits} cache hits"
        )
        return results

    def prepare_embedding_data(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self.generate_batch_embeddings(chunks)