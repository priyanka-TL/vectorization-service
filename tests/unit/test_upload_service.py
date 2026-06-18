"""
Unit tests for app/services/document_operations/upload_service.py

Focus on changes in this PR:
- _create_point_vectors: new sparse_vector parameter — included when provided, absent when None
- sparse_vector is stored under settings.SPARSE_VECTOR_NAME key
"""
import os
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


def _make_service():
    """Return an UploadService instance without importing the full app."""
    from app.services.document_operations.upload_service import UploadService
    with patch("app.services.document_operations.upload_service.embedding_model"):
        svc = UploadService.__new__(UploadService)
    return svc


class TestCreatePointVectors:
    """Tests for UploadService._create_point_vectors()."""

    def _text_embedding(self, dim=4):
        """Return a small numpy array mimicking a text embedding."""
        return np.array([0.1, 0.2, 0.3, 0.4])

    def test_returns_text_vector_always(self):
        svc = _make_service()
        embedding = self._text_embedding()
        result = svc._create_point_vectors(embedding, {})
        assert "text" in result
        assert result["text"] == embedding.tolist()

    def test_field_embeddings_included(self):
        svc = _make_service()
        embedding = self._text_embedding()
        title_emb = np.array([0.5, 0.6, 0.7, 0.8])
        field_embeddings = {"title": title_emb}
        result = svc._create_point_vectors(embedding, field_embeddings)
        assert "title" in result
        assert result["title"] == title_emb.tolist()

    def test_all_field_embeddings_included(self):
        svc = _make_service()
        embedding = self._text_embedding()
        field_embeddings = {
            "title": np.ones(4) * 0.1,
            "summary": np.ones(4) * 0.2,
            "tags": np.ones(4) * 0.3,
            "metadata": np.ones(4) * 0.4,
        }
        result = svc._create_point_vectors(embedding, field_embeddings)
        for field in ("title", "summary", "tags", "metadata"):
            assert field in result

    def test_no_sparse_vector_key_when_sparse_is_none(self):
        """When sparse_vector=None, SPARSE_VECTOR_NAME must not appear in the dict."""
        svc = _make_service()
        embedding = self._text_embedding()
        result = svc._create_point_vectors(embedding, {}, sparse_vector=None)
        from app.config import settings
        assert settings.SPARSE_VECTOR_NAME not in result

    def test_sparse_vector_included_when_provided(self):
        """When sparse_vector is provided, it must be stored under SPARSE_VECTOR_NAME."""
        svc = _make_service()
        embedding = self._text_embedding()
        fake_sparse = Mock()  # SparseVector model instance
        result = svc._create_point_vectors(embedding, {}, sparse_vector=fake_sparse)
        from app.config import settings
        assert settings.SPARSE_VECTOR_NAME in result
        assert result[settings.SPARSE_VECTOR_NAME] is fake_sparse

    def test_sparse_vector_not_included_by_default(self):
        """Default call (no sparse_vector arg) must not add sparse key."""
        svc = _make_service()
        embedding = self._text_embedding()
        result = svc._create_point_vectors(embedding, {})
        from app.config import settings
        assert settings.SPARSE_VECTOR_NAME not in result

    def test_sparse_vector_name_from_settings(self):
        """The key used for the sparse vector must come from settings.SPARSE_VECTOR_NAME."""
        svc = _make_service()
        embedding = self._text_embedding()
        fake_sparse = Mock()

        with patch("app.services.document_operations.upload_service.settings") as mock_settings:
            mock_settings.SPARSE_VECTOR_NAME = "my_bm25"
            mock_settings.SPARSE_SEARCH_ENABLED = True
            result = svc._create_point_vectors(embedding, {}, sparse_vector=fake_sparse)

        assert "my_bm25" in result
        assert result["my_bm25"] is fake_sparse

    def test_unknown_field_embeddings_are_ignored(self):
        """Fields not in ['title','summary','tags','metadata'] should not appear."""
        svc = _make_service()
        embedding = self._text_embedding()
        field_embeddings = {
            "title": np.ones(4),
            "unknown_field": np.ones(4),  # should be ignored
        }
        result = svc._create_point_vectors(embedding, field_embeddings)
        assert "unknown_field" not in result
        assert "title" in result

    def test_returns_dict(self):
        svc = _make_service()
        embedding = self._text_embedding()
        result = svc._create_point_vectors(embedding, {})
        assert isinstance(result, dict)

    def test_sparse_vector_alongside_dense_vectors(self):
        """Sparse and dense vectors can coexist in the returned dict."""
        svc = _make_service()
        embedding = self._text_embedding()
        title_emb = np.array([0.9, 0.8, 0.7, 0.6])
        fake_sparse = Mock()

        result = svc._create_point_vectors(
            embedding,
            {"title": title_emb},
            sparse_vector=fake_sparse,
        )
        assert "text" in result
        assert "title" in result
        from app.config import settings
        assert settings.SPARSE_VECTOR_NAME in result
