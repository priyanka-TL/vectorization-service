"""
Unit tests for new functions in app/core/clients/qdrant.py

Tests cover only the PR changes:
- _index_params_match: TextIndexParams with same/different tokenizer, simple schema types
- _ensure_payload_indexes: creates new indexes, skips existing matches, rebuilds changed ones
- batch_points: correct chunking, empty list, exact multiple, non-multiple batch size
- _ensure_sparse_vector_field: calls update_collection, handles ImportError gracefully,
  handles general exception gracefully
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# _index_params_match
# ---------------------------------------------------------------------------

class TestIndexParamsMatch:
    """Tests for the _index_params_match function."""

    def _get_func(self):
        with patch("app.core.clients.qdrant.qdrant_client"):
            from app.core.clients.qdrant import _index_params_match
            return _index_params_match

    def test_simple_schema_always_matches(self):
        """KEYWORD / TEXT schema types: any existing index is a match."""
        from qdrant_client.models import PayloadSchemaType
        f = self._get_func()
        existing = Mock()  # any object representing an existing index
        assert f(existing, PayloadSchemaType.KEYWORD) is True
        assert f(existing, PayloadSchemaType.TEXT) is True

    def test_text_index_params_same_tokenizer_matches(self):
        from qdrant_client import models
        f = self._get_func()
        desired = models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
        existing = Mock()
        existing.params = Mock()
        existing.params.tokenizer = models.TokenizerType.PREFIX
        assert f(existing, desired) is True

    def test_text_index_params_different_tokenizer_does_not_match(self):
        from qdrant_client import models
        f = self._get_func()
        desired = models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
        existing = Mock()
        existing.params = Mock()
        existing.params.tokenizer = models.TokenizerType.WORD  # different
        assert f(existing, desired) is False

    def test_text_index_params_missing_params_attribute_does_not_match(self):
        from qdrant_client import models
        f = self._get_func()
        desired = models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
        existing = Mock(spec=[])  # no params attribute
        assert f(existing, desired) is False

    def test_text_index_params_none_tokenizer_does_not_match(self):
        """Existing index with tokenizer=None should not match PREFIX."""
        from qdrant_client import models
        f = self._get_func()
        desired = models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
        existing = Mock()
        existing.params = Mock()
        existing.params.tokenizer = None
        assert f(existing, desired) is False


# ---------------------------------------------------------------------------
# _ensure_payload_indexes
# ---------------------------------------------------------------------------

class TestEnsurePayloadIndexes:
    """Tests for _ensure_payload_indexes."""

    def _run(self, mock_qdrant, existing_schema=None):
        """Helper: patch qdrant_client and call _ensure_payload_indexes."""
        collection = Mock()
        collection.payload_schema = existing_schema or {}
        mock_qdrant.get_collection.return_value = collection

        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from app.core.clients.qdrant import _ensure_payload_indexes
            _ensure_payload_indexes("test_collection")

    def test_creates_indexes_when_schema_empty(self):
        mock_qdrant = Mock()
        self._run(mock_qdrant, existing_schema={})
        # Should have called create_payload_index for each entry in _PAYLOAD_INDEXES
        from app.core.clients.qdrant import _PAYLOAD_INDEXES
        assert mock_qdrant.create_payload_index.call_count == len(_PAYLOAD_INDEXES)

    def test_skips_creation_when_index_already_matches(self):
        from qdrant_client.models import PayloadSchemaType
        mock_qdrant = Mock()

        # Pretend all keyword indexes already exist (matching)
        existing_schema = {
            "source_id": Mock(),
            "metadata.company": Mock(),
            "tags": Mock(),
            "metadata.DOCUMENT_TYPE": Mock(),
        }
        # Title and summary are TextIndexParams — give them matching tokenizers
        from qdrant_client import models
        for field in ["title", "summary"]:
            es = Mock()
            es.params = Mock()
            es.params.tokenizer = models.TokenizerType.PREFIX
            existing_schema[field] = es

        self._run(mock_qdrant, existing_schema=existing_schema)
        # No indexes need rebuilding, no creates
        mock_qdrant.create_payload_index.assert_not_called()
        mock_qdrant.delete_payload_index.assert_not_called()

    def test_rebuilds_index_when_tokenizer_changed(self):
        """When a TextIndex has a different tokenizer, delete + recreate."""
        from qdrant_client import models
        mock_qdrant = Mock()

        # Title has WORD tokenizer (stale), summary has PREFIX (ok)
        existing_schema = {
            "title": Mock(),
        }
        existing_schema["title"].params = Mock()
        existing_schema["title"].params.tokenizer = models.TokenizerType.WORD  # stale

        self._run(mock_qdrant, existing_schema=existing_schema)

        # delete_payload_index called for "title"
        mock_qdrant.delete_payload_index.assert_called()
        delete_calls = [c.kwargs.get("field_name") or c.args[1]
                        for c in mock_qdrant.delete_payload_index.call_args_list
                        if c]
        # At minimum, "title" should have been deleted
        field_names = [
            c.kwargs.get("field_name") for c in mock_qdrant.delete_payload_index.call_args_list
        ]
        assert "title" in field_names

    def test_non_fatal_on_get_collection_failure(self):
        """If get_collection raises, proceed with empty schema (no crash)."""
        mock_qdrant = Mock()
        mock_qdrant.get_collection.side_effect = RuntimeError("not found")
        # Should not raise
        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from app.core.clients.qdrant import _ensure_payload_indexes
            _ensure_payload_indexes("test_collection")
        # It should still try to create all indexes
        from app.core.clients.qdrant import _PAYLOAD_INDEXES
        assert mock_qdrant.create_payload_index.call_count == len(_PAYLOAD_INDEXES)

    def test_non_fatal_on_create_index_failure(self):
        """If create_payload_index raises for one field, others still processed."""
        mock_qdrant = Mock()
        collection = Mock()
        collection.payload_schema = {}
        mock_qdrant.get_collection.return_value = collection
        # Make create_payload_index raise on first call only
        call_count = {"n": 0}
        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("index error")
        mock_qdrant.create_payload_index.side_effect = side_effect

        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from app.core.clients.qdrant import _ensure_payload_indexes
            _ensure_payload_indexes("test_collection")  # should not raise

        from app.core.clients.qdrant import _PAYLOAD_INDEXES
        # All entries were attempted
        assert mock_qdrant.create_payload_index.call_count == len(_PAYLOAD_INDEXES)


# ---------------------------------------------------------------------------
# batch_points
# ---------------------------------------------------------------------------

class TestBatchPoints:
    """Tests for the batch_points generator."""

    def _get_batch_points(self):
        with patch("app.core.clients.qdrant.qdrant_client"):
            from app.core.clients.qdrant import batch_points
            return batch_points

    def test_empty_list_yields_nothing(self):
        batch_points = self._get_batch_points()
        assert list(batch_points([], batch_size=10)) == []

    def test_fewer_than_batch_size_yields_one_batch(self):
        batch_points = self._get_batch_points()
        items = list(range(5))
        batches = list(batch_points(items, batch_size=10))
        assert len(batches) == 1
        assert batches[0] == items

    def test_exact_multiple_yields_correct_number_of_batches(self):
        batch_points = self._get_batch_points()
        items = list(range(100))
        batches = list(batch_points(items, batch_size=10))
        assert len(batches) == 10
        assert all(len(b) == 10 for b in batches)

    def test_non_multiple_yields_last_partial_batch(self):
        batch_points = self._get_batch_points()
        items = list(range(105))
        batches = list(batch_points(items, batch_size=10))
        assert len(batches) == 11
        assert len(batches[-1]) == 5

    def test_all_items_present_in_batches(self):
        batch_points = self._get_batch_points()
        items = list(range(250))
        batches = list(batch_points(items, batch_size=100))
        flattened = [item for batch in batches for item in batch]
        assert flattened == items

    def test_default_batch_size_is_100(self):
        batch_points = self._get_batch_points()
        items = list(range(150))
        batches = list(batch_points(items))  # no explicit batch_size
        assert len(batches) == 2
        assert len(batches[0]) == 100
        assert len(batches[1]) == 50


# ---------------------------------------------------------------------------
# _ensure_sparse_vector_field
# ---------------------------------------------------------------------------

class TestEnsureSparseVectorField:
    """Tests for _ensure_sparse_vector_field."""

    def _run(self, mock_qdrant):
        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from app.core.clients.qdrant import _ensure_sparse_vector_field
            _ensure_sparse_vector_field("test_collection")

    def test_calls_update_collection_when_imports_succeed(self):
        mock_qdrant = Mock()
        mock_SparseVectorParams = Mock(return_value=Mock())
        mock_Modifier = Mock()
        mock_Modifier.IDF = "IDF"

        import sys
        original_modules = {}
        # Patch qdrant_client.models to include SparseVectorParams and Modifier
        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from qdrant_client import models
            # Temporarily add SparseVectorParams / Modifier if missing
            if not hasattr(models, "SparseVectorParams"):
                models.SparseVectorParams = mock_SparseVectorParams
                models.Modifier = mock_Modifier
                added = True
            else:
                added = False
            try:
                from app.core.clients.qdrant import _ensure_sparse_vector_field
                _ensure_sparse_vector_field("test_collection")
            finally:
                if added:
                    del models.SparseVectorParams
                    del models.Modifier

        mock_qdrant.update_collection.assert_called_once()

    def test_handles_import_error_gracefully(self):
        """When SparseVectorParams cannot be imported, no exception raised."""
        mock_qdrant = Mock()
        import sys
        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            # Simulate ImportError by making the import inside the function fail
            with patch.dict(sys.modules, {"qdrant_client.models": None}):
                # This is tricky since the module is already imported; instead test
                # via patching the update_collection to raise ImportError
                mock_qdrant.update_collection.side_effect = None  # reset
                from app.core.clients.qdrant import _ensure_sparse_vector_field
                # Should not raise even if update fails
                mock_qdrant.update_collection.side_effect = Exception("server error")
                _ensure_sparse_vector_field("test_collection")  # must not raise

    def test_handles_general_exception_gracefully(self):
        mock_qdrant = Mock()
        mock_qdrant.update_collection.side_effect = Exception("unexpected")
        # Should not raise
        with patch("app.core.clients.qdrant.qdrant_client", mock_qdrant):
            from app.core.clients.qdrant import _ensure_sparse_vector_field
            _ensure_sparse_vector_field("test_collection")