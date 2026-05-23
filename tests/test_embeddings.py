# tests/test_embeddings.py
"""
Unit tests for EmbeddingGenerator.
All OpenAI API calls are mocked — no API key or credits required.
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock


FAKE_EMBEDDING = [0.1] * 1024


def _make_mock_response():
    resp = MagicMock()
    resp.data = [MagicMock(embedding=FAKE_EMBEDDING)]
    return resp


@pytest.fixture
def generator(tmp_path, monkeypatch):
    """Return an EmbeddingGenerator with mocked OpenAI client and temp cache."""
    monkeypatch.setenv("NVIDIA_EMBEDDING_API_KEY", "nvapi-test-key")

    cache_file = tmp_path / "embedding_cache.json"
    monkeypatch.setattr("backend.embeddings.EMBEDDING_CACHE_FILE", cache_file)

    with patch("backend.embeddings.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_mock_response()
        MockOpenAI.return_value = mock_client

        from backend.embeddings import EmbeddingGenerator
        gen = EmbeddingGenerator()
        gen.client = mock_client   # ensure the instance uses the mock
        yield gen


class TestGenerateEmbedding:
    def test_returns_correct_dimension(self, generator):
        result = generator.generate_embedding("test text")
        assert len(result) == 1024

    def test_returns_list_of_floats(self, generator):
        result = generator.generate_embedding("test text")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_cache_hit_skips_api(self, generator):
        generator.generate_embedding("cached text")
        first_call_count = generator.client.embeddings.create.call_count
        generator.generate_embedding("cached text")   # should hit cache
        assert generator.client.embeddings.create.call_count == first_call_count

    def test_different_texts_call_api_separately(self, generator):
        generator.generate_embedding("text one")
        generator.generate_embedding("text two")
        assert generator.client.embeddings.create.call_count == 2


class TestBatchEmbeddings:
    def test_enriches_chunks_with_embedding(self, generator):
        chunks = [
            {"text": "chunk one", "source_file": "a.pdf",
             "page_number": 1, "chunk_index": 0, "token_count": 5,
             "char_count": 9, "word_count": 2, "chunk_id": "id1"},
        ]
        results = generator.generate_batch_embeddings(chunks)
        assert len(results) == 1
        assert "embedding" in results[0]
        assert "timestamp" in results[0]
        # Original metadata must be preserved
        assert results[0]["source_file"] == "a.pdf"
        assert results[0]["page_number"] == 1

    def test_all_chunks_get_embeddings(self, generator):
        chunks = [
            {"text": f"chunk {i}", "source_file": "a.pdf",
             "page_number": i, "chunk_index": i, "token_count": 2,
             "char_count": 7, "word_count": 2, "chunk_id": f"id{i}"}
            for i in range(5)
        ]
        results = generator.generate_batch_embeddings(chunks)
        assert len(results) == 5
        assert all("embedding" in r for r in results)


class TestDiskCache:
    def test_cache_size_property(self, generator):
        generator.generate_embedding("new text abc")
        assert generator.cache_size >= 1

    def test_clear_cache(self, generator):
        generator.generate_embedding("something")
        generator.clear_cache()
        assert generator.cache_size == 0

    def test_cache_persists_to_disk(self, generator, tmp_path, monkeypatch):
        cache_file = tmp_path / "embedding_cache.json"
        monkeypatch.setattr("backend.embeddings.EMBEDDING_CACHE_FILE", cache_file)
        generator.generate_embedding("persist this")
        generator._save_disk_cache()
        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)
        assert len(data) >= 1