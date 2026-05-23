# tests/test_text_chunker.py
"""Unit tests for TextChunker — no OpenAI calls needed (uses tiktoken locally)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.text_chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(default_chunk_size=200, default_overlap=20)


SAMPLE_TEXT = (
    "Artificial intelligence is transforming every industry. "
    "Machine learning models are being deployed at scale. "
    "Natural language processing enables computers to understand human speech. "
    "Deep learning has revolutionised image recognition tasks. "
    "Reinforcement learning allows agents to learn from their environment. "
) * 10   # ~500 tokens


class TestCountTokens:
    def test_non_zero_for_text(self, chunker):
        assert chunker.count_tokens("hello world") > 0

    def test_empty_string(self, chunker):
        assert chunker.count_tokens("") == 0


class TestChunkByTokens:
    def test_returns_list(self, chunker):
        chunks = chunker.chunk_by_tokens(SAMPLE_TEXT, "doc.pdf", 1)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_all_chunks_have_required_keys(self, chunker):
        required = {"chunk_id", "chunk_index", "text", "source_file",
                    "page_number", "token_count"}
        for chunk in chunker.chunk_by_tokens(SAMPLE_TEXT, "doc.pdf", 1):
            assert required.issubset(chunk.keys())

    def test_chunk_ids_are_unique(self, chunker):
        chunks = chunker.chunk_by_tokens(SAMPLE_TEXT, "doc.pdf", 1)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_returns_empty(self, chunker):
        assert chunker.chunk_by_tokens("", "doc.pdf", 1) == []

    def test_source_file_propagated(self, chunker):
        chunks = chunker.chunk_by_tokens(SAMPLE_TEXT, "myfile.pdf", 3)
        assert all(c["source_file"] == "myfile.pdf" for c in chunks)

    def test_page_number_propagated(self, chunker):
        chunks = chunker.chunk_by_tokens(SAMPLE_TEXT, "doc.pdf", 7)
        assert all(c["page_number"] == 7 for c in chunks)


class TestChunkBySentences:
    def test_returns_chunks(self, chunker):
        chunks = chunker.chunk_by_sentences(SAMPLE_TEXT, "doc.pdf", 1)
        assert len(chunks) > 0

    def test_empty_text(self, chunker):
        assert chunker.chunk_by_sentences("", "doc.pdf", 1) == []


class TestMergeSmallChunks:
    def test_merges_tiny_chunks(self, chunker):
        tiny = [
            {"text": "hi", "token_count": 1, "char_count": 2, "word_count": 1,
             "chunk_id": f"x_{i}", "chunk_index": i,
             "source_file": "f.pdf", "page_number": 1}
            for i in range(5)
        ]
        merged = chunker.merge_small_chunks(tiny, min_tokens=10)
        assert len(merged) < len(tiny)

    def test_empty_input(self, chunker):
        assert chunker.merge_small_chunks([]) == []


class TestCreateChunks:
    def test_tokens_strategy(self, chunker):
        chunks = chunker.create_chunks(SAMPLE_TEXT, "doc.pdf", 1, strategy="tokens")
        assert len(chunks) > 0

    def test_sentences_strategy(self, chunker):
        chunks = chunker.create_chunks(SAMPLE_TEXT, "doc.pdf", 1, strategy="sentences")
        assert len(chunks) > 0

    def test_optimal_size_short_doc(self, chunker):
        short_text = "Short document. " * 10
        assert chunker.calculate_optimal_chunk_size(short_text) == 500

    def test_optimal_size_long_doc(self, chunker):
        long_text = "word " * 21000   # 21 000 tokens with cl100k_base → returns 1500
        assert chunker.calculate_optimal_chunk_size(long_text) == 1500