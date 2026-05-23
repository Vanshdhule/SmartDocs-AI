# tests/test_ingestion_pipeline.py
"""Integration tests for ingestion pipeline — workspace-aware. All API calls mocked."""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch, MagicMock

FAKE_EMB   = [0.01] * 1024
FAKE_PAGES = [
    {"page_number": 1, "text": "Artificial intelligence transforms industries."},
    {"page_number": 2, "text": "Machine learning enables computers to learn."},
]

def _make_file(name="test.pdf"):
    f = io.BytesIO(b"%PDF-1.4 fake"); f.name = name; f.size = 20; return f

def _mock_pipeline(tmp_path, monkeypatch, workspace="Default"):
    monkeypatch.setenv("NVIDIA_EMBEDDING_API_KEY", "nvapi-test")
    monkeypatch.setattr("backend.embeddings.EMBEDDING_CACHE_FILE", tmp_path / "cache.json")
    ctx = [
        patch("backend.pdf_processor.PYMUPDF_AVAILABLE", True),
        patch("backend.pdf_processor.fitz"),
        patch("backend.embeddings.OpenAI"),
    ]
    for c in ctx: c.start()
    import fitz as mock_fitz_mod
    from unittest.mock import MagicMock as MM
    mock_doc = MM(); mock_doc.is_encrypted = False; mock_doc.page_count = 2
    mock_doc.metadata = {}; mock_doc.load_page.side_effect = lambda i: MM(get_text=lambda: FAKE_PAGES[i]["text"])
    from backend.pdf_processor import PDFProcessor
    import backend.pdf_processor as pdfmod
    pdfmod.fitz = MM(); pdfmod.fitz.open.return_value = mock_doc; pdfmod.fitz.FileDataError = Exception
    from backend.embeddings import EmbeddingGenerator
    import backend.embeddings as embmod
    mock_client = MM()
    mock_resp = MM(); mock_resp.data = [MM(embedding=FAKE_EMB)]
    mock_client.embeddings.create.return_value = mock_resp
    embmod.OpenAI.return_value = mock_client
    from backend.ingestion_pipeline import DocumentIngestion
    from backend.vector_db import VectorDatabase
    pip = DocumentIngestion(workspace=workspace)
    pip.vector_db = VectorDatabase(persist_dir=str(tmp_path / f"ch_{workspace}"), workspace=workspace)
    pip.embedding_generator.client = mock_client
    return pip

@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    return _mock_pipeline(tmp_path, monkeypatch, "Default")

@pytest.fixture
def pipeline_ws(tmp_path, monkeypatch):
    return _mock_pipeline(tmp_path, monkeypatch, "Test WS")

class TestProcessSingleDocument:
    def test_success_result_shape(self, pipeline):
        r = pipeline.process_single_document(_make_file())
        assert r["success"] is True
        assert r["file_name"] == "test.pdf"
        assert r["chunks"] >= 1
        assert r["already_indexed"] is False

    def test_workspace_in_result(self, pipeline):
        r = pipeline.process_single_document(_make_file())
        assert r["workspace"] == "Default"

    def test_chunks_stored_in_chroma(self, pipeline):
        pipeline.process_single_document(_make_file())
        assert pipeline.vector_db.get_collection_stats()["count"] >= 1

    def test_dedup_skip_on_second_upload(self, pipeline):
        f = _make_file()
        pipeline.process_single_document(f)
        f.seek(0)
        assert pipeline.process_single_document(f)["already_indexed"] is True

    def test_dedup_replace_keeps_same_count(self, pipeline):
        f = _make_file()
        pipeline.process_single_document(f)
        before = pipeline.vector_db.get_collection_stats()["count"]
        f.seek(0)
        pipeline.process_single_document(f, replace_existing=True)
        assert pipeline.vector_db.get_collection_stats()["count"] == before

    def test_tokens_strategy(self, pipeline):
        assert pipeline.process_single_document(_make_file("tok.pdf"), chunk_strategy="tokens")["success"]

    def test_sentences_strategy(self, pipeline):
        assert pipeline.process_single_document(_make_file("sent.pdf"), chunk_strategy="sentences")["success"]

class TestWorkspaceIsolation:
    def test_different_workspaces_isolated(self, pipeline, pipeline_ws):
        pipeline.process_single_document(_make_file("shared.pdf"))
        assert pipeline_ws.vector_db.get_collection_stats()["count"] == 0

    def test_switch_workspace(self, pipeline):
        pipeline.switch_workspace("Switched WS")
        assert pipeline.workspace == "Switched WS"
        assert pipeline.vector_db.workspace_name == "Switched WS"

class TestRollback:
    def test_rollback_cleans_partial_inserts(self, pipeline):
        fake_ids = ["rb_1", "rb_2"]
        pipeline.vector_db.collection.add(
            ids=fake_ids, embeddings=[FAKE_EMB, FAKE_EMB],
            documents=["d1","d2"],
            metadatas=[{"source_file":"x.pdf","page_number":1,"chunk_index":0,"token_count":5,"workspace":"Default"},
                       {"source_file":"x.pdf","page_number":1,"chunk_index":1,"token_count":5,"workspace":"Default"}],
        )
        assert pipeline.vector_db.get_collection_stats()["count"] == 2
        pipeline.rollback_on_error(fake_ids)
        assert pipeline.vector_db.get_collection_stats()["count"] == 0