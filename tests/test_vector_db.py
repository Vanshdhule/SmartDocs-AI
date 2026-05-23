# tests/test_vector_db.py
"""Unit tests for VectorDatabase — workspace-aware. Uses real ephemeral ChromaDB."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

FAKE_EMB = [0.1] * 1024  # 1024-dim: NVIDIA nv-embedqa-e5-v5

def make_chunk(chunk_id="chunk_001", source="test.pdf", page=1, idx=0):
    return {"chunk_id": chunk_id, "text": f"Sample text for chunk {idx}",
            "source_file": source, "page_number": page, "chunk_index": idx,
            "token_count": 10, "embedding": FAKE_EMB}

@pytest.fixture
def db(tmp_path):
    from backend.vector_db import VectorDatabase
    return VectorDatabase(persist_dir=str(tmp_path / "chroma_test"))

@pytest.fixture
def db_ws(tmp_path):
    from backend.vector_db import VectorDatabase
    return VectorDatabase(persist_dir=str(tmp_path / "chroma_ws"), workspace="Test Workspace")

class TestConnection:
    def test_verify_connection(self, db):
        assert db.verify_connection() is True
    def test_collection_exists_after_init(self, db):
        assert db.check_collection_exists() is True

class TestWorkspace:
    def test_default_workspace_name(self, db):
        assert db.workspace_name == "Default"
    def test_custom_workspace_name(self, db_ws):
        assert db_ws.workspace_name == "Test Workspace"
    def test_collection_name_prefixed(self, db):
        assert db.collection_name.startswith("smartdocs_")
    def test_switch_workspace_changes_collection(self, db):
        original = db.collection_name
        db.set_workspace("New Workspace")
        assert db.collection_name != original
        assert db.workspace_name == "New Workspace"
    def test_workspace_isolation(self, tmp_path):
        from backend.vector_db import VectorDatabase
        db_a = VectorDatabase(persist_dir=str(tmp_path / "ch"), workspace="WS-A")
        db_b = VectorDatabase(persist_dir=str(tmp_path / "ch"), workspace="WS-B")
        db_a.add_documents([make_chunk(chunk_id="only_in_a")])
        assert db_b.get_collection_stats()["count"] == 0
    def test_delete_workspace(self, tmp_path):
        from backend.vector_db import VectorDatabase
        db = VectorDatabase(persist_dir=str(tmp_path / "ch"), workspace="ToDelete")
        db.add_documents([make_chunk()])
        db.delete_workspace("ToDelete")
        db2 = VectorDatabase(persist_dir=str(tmp_path / "ch"), workspace="ToDelete")
        assert db2.get_collection_stats()["count"] == 0
    def test_get_workspace_doc_stats(self, db):
        db.add_documents([make_chunk(source="report.pdf", page=1)])
        db.add_documents([make_chunk(chunk_id="id2", source="report.pdf", page=2)])
        stats = db.get_workspace_doc_stats()
        assert len(stats) == 1
        assert stats[0]["name"] == "report.pdf"
        assert stats[0]["chunks"] == 2
    def test_get_workspace_docs_returns_names(self, db):
        db.add_documents([make_chunk(source="doc_a.pdf")])
        db.add_documents([make_chunk(chunk_id="id2", source="doc_b.pdf")])
        docs = db.get_workspace_docs()
        assert "doc_a.pdf" in docs and "doc_b.pdf" in docs

class TestAddAndQuery:
    def test_add_documents_increases_count(self, db):
        db.add_documents([make_chunk()])
        assert db.get_collection_stats()["count"] == 1
    def test_chunk_id_used_as_document_id(self, db):
        db.add_documents([make_chunk(chunk_id="my-specific-id")])
        assert "my-specific-id" in db.collection.get(ids=["my-specific-id"])["ids"]
    def test_query_returns_results(self, db):
        db.add_documents([make_chunk()])
        assert len(db.query_documents(FAKE_EMB, n_results=1)["ids"][0]) == 1
    def test_query_metadata_correct(self, db):
        db.add_documents([make_chunk(source="myfile.pdf", page=3)])
        meta = db.query_documents(FAKE_EMB, n_results=1)["metadatas"][0][0]
        assert meta["source_file"] == "myfile.pdf" and meta["page_number"] == 3
    def test_workspace_stored_in_metadata(self, db_ws):
        db_ws.add_documents([make_chunk()])
        meta = db_ws.query_documents(FAKE_EMB, n_results=1)["metadatas"][0][0]
        assert meta.get("workspace") == "Test Workspace"

class TestGetSourceFileIds:
    def test_returns_ids_for_known_file(self, db):
        db.add_documents([make_chunk(chunk_id="id_a", source="known.pdf")])
        assert "id_a" in db.get_source_file_ids("known.pdf")
    def test_returns_empty_for_unknown_file(self, db):
        assert db.get_source_file_ids("nonexistent.pdf") == []
    def test_returns_all_chunks_for_file(self, db):
        db.add_documents([make_chunk(chunk_id=f"id_{i}", source="multi.pdf", idx=i) for i in range(3)])
        assert len(db.get_source_file_ids("multi.pdf")) == 3

class TestDeleteDocuments:
    def test_delete_reduces_count(self, db):
        db.add_documents([make_chunk(chunk_id="del_me")])
        db.delete_documents(["del_me"])
        assert db.get_collection_stats()["count"] == 0
    def test_delete_nonexistent_does_not_raise(self, db):
        db.delete_documents(["ghost_id"])

class TestCollectionStats:
    def test_stats_has_required_keys(self, db):
        stats = db.get_collection_stats()
        for key in ["count", "collection_name", "is_empty", "workspace"]:
            assert key in stats
    def test_is_empty_true_on_fresh_db(self, db):
        assert db.get_collection_stats()["is_empty"] is True
    def test_workspace_in_stats(self, db_ws):
        assert db_ws.get_collection_stats()["workspace"] == "Test Workspace"