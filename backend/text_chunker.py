# backend/text_chunker.py
"""
Text Chunking Module
Splits cleaned text into overlapping chunks for embedding and retrieval.
Supports token-based and sentence-based strategies.
"""

import hashlib
import logging
from typing import List, Dict, Optional

import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

try:
    from config import (
        CHAT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    )
except ImportError:
    CHAT_MODEL           = "gpt-3.5-turbo"
    DEFAULT_CHUNK_SIZE   = 1000
    DEFAULT_CHUNK_OVERLAP = 200

logger = logging.getLogger("smartdocs.text_chunker")

# Download punkt tokenizer data quietly on first import
nltk.download("punkt_tab", quiet=True)


class TextChunker:
    def __init__(
        self,
        model_name: str = CHAT_MODEL,
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
        default_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Mistral and other non-OpenAI models are not in tiktoken's registry.
            # cl100k_base (GPT-4 tokeniser) is a safe, accurate fallback.
            logger.debug(
                f"tiktoken has no mapping for '{model_name}' — using cl100k_base"
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.default_chunk_size = default_chunk_size
        self.default_overlap    = default_overlap

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def calculate_optimal_chunk_size(self, text: str) -> int:
        """Scale chunk size to document length."""
        total = self.count_tokens(text)
        if total < 3_000:   return 500
        if total < 20_000:  return 1_000
        return 1_500

    def merge_small_chunks(
        self, chunks: List[Dict], min_tokens: int = 300
    ) -> List[Dict]:
        """Merge under-sized chunks with their neighbour."""
        if not chunks:
            return []
        merged = []
        buf = chunks[0]
        for chunk in chunks[1:]:
            if buf["token_count"] < min_tokens:
                buf["text"]        += " " + chunk["text"]
                buf["token_count"] += chunk["token_count"]
                buf["char_count"]  += chunk["char_count"]
                buf["word_count"]  += chunk["word_count"]
            else:
                merged.append(buf)
                buf = chunk
        merged.append(buf)
        return merged

    # ── Strategies ────────────────────────────────────────────────────────────

    def chunk_by_tokens(
        self,
        text: str,
        source_file: str,
        page_number: int,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[Dict]:
        """Fixed-size token windows with overlap."""
        if not text.strip():
            return []
        chunk_size = chunk_size or self.default_chunk_size
        overlap    = overlap    or self.default_overlap
        tokens     = self.encoding.encode(text)
        chunks, start, idx = [], 0, 0
        while start < len(tokens):
            end        = start + chunk_size
            chunk_text = self.encoding.decode(tokens[start:end])
            chunks.append(self._build_chunk(chunk_text, idx, source_file, page_number))
            start = end - overlap
            idx  += 1
        return chunks

    def chunk_by_sentences(
        self,
        text: str,
        source_file: str,
        page_number: int,
        max_tokens: Optional[int] = None,
        overlap_sentences: int = 2,
    ) -> List[Dict]:
        """Sentence-boundary-aware chunking."""
        if not text.strip():
            return []
        max_tokens         = max_tokens or self.default_chunk_size
        sentences          = sent_tokenize(text)
        chunks             = []
        current_sentences  = []
        current_tokens     = 0
        idx                = 0
        for sentence in sentences:
            s_tokens = self.count_tokens(sentence)
            if current_tokens + s_tokens > max_tokens and current_sentences:
                chunks.append(
                    self._build_chunk(
                        " ".join(current_sentences), idx, source_file, page_number
                    )
                )
                current_sentences = current_sentences[-overlap_sentences:]
                current_tokens    = self.count_tokens(" ".join(current_sentences))
                idx += 1
            current_sentences.append(sentence)
            current_tokens += s_tokens
        if current_sentences:
            chunks.append(
                self._build_chunk(
                    " ".join(current_sentences), idx, source_file, page_number
                )
            )
        return chunks

    # ── Master entry point ────────────────────────────────────────────────────

    def create_chunks(
        self,
        text: str,
        source_file: str,
        page_number: int,
        strategy: str = "tokens",
    ) -> List[Dict]:
        optimal = self.calculate_optimal_chunk_size(text)
        if strategy == "sentences":
            chunks = self.chunk_by_sentences(
                text, source_file, page_number, max_tokens=optimal
            )
        else:
            chunks = self.chunk_by_tokens(
                text, source_file, page_number, chunk_size=optimal
            )
        return self.merge_small_chunks(chunks)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_doc_id(self, source_file: str) -> str:
        return hashlib.md5(source_file.encode()).hexdigest()[:12]

    def _build_chunk(
        self, text: str, chunk_index: int, source_file: str, page_number: int
    ) -> Dict:
        doc_id = self._generate_doc_id(source_file)
        return {
            "chunk_id":    f"{doc_id}_{chunk_index}",
            "chunk_index": chunk_index,
            "text":        text,
            "source_file": source_file,
            "page_number": page_number,
            "token_count": self.count_tokens(text),
            "char_count":  len(text),
            "word_count":  len(text.split()),
        }