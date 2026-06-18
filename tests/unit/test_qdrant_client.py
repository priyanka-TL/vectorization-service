"""
Unit tests for app/core/clients/qdrant.py

Focus on functions added/changed in this PR:
- _index_params_match: TextIndexParams comparison vs simple schema types
- _ensure_payload_indexes: creates, rebuilds, and skips indexes correctly
- _ensure_sparse_vector_field: calls update_collection, handles ImportError/exceptions
- batch_points: correct chunking behaviour
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_text_index_params(tokenizer_type):
    """Build a models.TextIndexParams with the given tokenizer."""
    from qdrant_client import models
    return models.TextIndexParams(
        type="text",
        tokenizer=tokenizer_type,
        min_token_len=2,
        max_token_len=20,
        lowercase=True,
    )


# ---------------------------------------------------------------------------
# _index_params_match
# ---------------------------------------------------------------------------

class TestIndexParamsMatch:
    """Tests for _index_params_match()."""

    def test_simple_keyword_schema_always_matches(self):
        """For KEYWORD schema, any existing index is treated as a match."""
        from app.core.clients.qdrant import _index_params_match
        from qdrant_client.models import PayloadSchemaType

        existing = Mock()  # any object — presence is enough
        assert _index_params_match(existing, PayloadSchemaType.KEYWORD) is True

    def test_simple_text_schema_always_matches(self):
        """For plain TEXT schema type, any existing index is a match."""
        from app.core.clients.qdrant import _index_params_match
        from qdrant_client.models import PayloadSchemaType

        existing = Mock()
        assert _index_params_match(existing, PayloadSchemaType.TEXT) is True

    def test_text_index_params_match_when_tokenizer_same(self):
        """TextIndexParams: match when existing tokenizer == desired tokenizer."""
        from app.core.clients.qdrant import _index_params_match
        from qdrant_client import models

        desired = _make_text_index_params(models.TokenizerType.PREFIX)

        existing_params = Mock()
        existing_params.tokenizer = models.TokenizerType.PREFIX
        existing = Mock()
        existing.params = existing_params

        assert _index_params_match(existing, desired) is True

    def test_text_index_params_no_match_when_tokenizer_differs(self):
        """TextIndexParams: no match when existing tokenizer != desired tokenizer."""
        from app.core.clients.qdrant import _index_params_match
        from qdrant_client import models

        desired = _make_text_index_params(models.TokenizerType.PREFIX)

        existing_params = Mock()
        existing_params.tokenizer = models.TokenizerType.WORD  # different
        existing = Mock()
        existing.params = existing_params

        assert _index_params_match(existing, desired) is False

    def test_text_index_params_no_match_when_existing_has_no_params(self):
        """TextIndexParams: no match when existing has no 'params' attribute."""
        from app.core.clients.qdrant import _index_params_match
        from qdrant_client import models

        desired = _make_text_index_params(models.TokenizerType.PREFIX)
        existing = Mock(spec=[])  # no attributes at all

        assert _index_params_match(existing, desired) is False


# ---------------------------------------------------------------------------
# _ensure_payload_indexes
# ---------------------------------------------------------------------------

class TestEnsurePayloadIndexes:
    """Tests for _ensure_payload_indexes()."""

    def _make_mock_client(self, payload_schema=None):
        """Return a mock qdrant_client with get_collection configured."""
        mock_collection = Mock()
        mock_collection.payload_schema = payload_schema or {}
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        return mock_client

    def test_creates_all_indexes_when_none_exist(self):
        """All _PAYLOAD_INDEXES entries should be created when schema is empty."""
        from app.core.clients import qdrant as qdrant_mod
        from app.core.clients.qdrant import _PAYLOAD_INDEXES

        mock_client = self._make_mock_client(payload_schema={})

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            qdrant_mod._ensure_payload_indexes("test_collection")

        assert mock_client.create_payload_index.call_count == len(_PAYLOAD_INDEXES)
        mock_client.delete_payload_index.assert_not_called()

    def test_skips_existing_indexes_that_match(self):
        """Indexes that already exist and match should not be recreated."""
        from app.core.clients import qdrant as qdrant_mod
        from qdrant_client.models import PayloadSchemaType

        # Simulate source_id already indexed as KEYWORD (matches desired)
        existing_source_id = Mock()
        # _index_params_match for simple schema always returns True
        schema = {"source_id": existing_source_id}
        mock_client = self._make_mock_client(payload_schema=schema)

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            qdrant_mod._ensure_payload_indexes("test_collection")

        # source_id should NOT be created (it already matches)
        created_fields = [
            c.kwargs.get("field_name") or c.args[1]
            for c in mock_client.create_payload_index.call_args_list
        ]
        assert "source_id" not in created_fields

    def test_rebuilds_stale_text_index(self):
        """When tokenizer changed, the old index is deleted and a new one created."""
        from app.core.clients import qdrant as qdrant_mod
        from qdrant_client import models

        # Simulate 'title' indexed with WORD tokenizer (stale)
        existing_params = Mock()
        existing_params.tokenizer = models.TokenizerType.WORD  # != PREFIX
        existing_title = Mock()
        existing_title.params = existing_params

        schema = {"title": existing_title}
        mock_client = self._make_mock_client(payload_schema=schema)

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            qdrant_mod._ensure_payload_indexes("test_collection")

        # delete then create for 'title'
        delete_calls = [c for c in mock_client.delete_payload_index.call_args_list]
        assert any("title" in str(c) for c in delete_calls)
        create_calls = [c for c in mock_client.create_payload_index.call_args_list]
        assert any("title" in str(c) for c in create_calls)

    def test_continues_on_individual_field_error(self):
        """Errors on individual field indexes are non-fatal; remaining fields proceed."""
        from app.core.clients import qdrant as qdrant_mod
        from app.core.clients.qdrant import _PAYLOAD_INDEXES

        mock_client = self._make_mock_client(payload_schema={})
        # First create_payload_index call raises; rest succeed
        call_count = [0]

        def create_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("transient error")

        mock_client.create_payload_index.side_effect = create_side_effect

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            # Should not raise
            qdrant_mod._ensure_payload_indexes("test_collection")

        # All fields still attempted even though first one failed
        assert mock_client.create_payload_index.call_count == len(_PAYLOAD_INDEXES)

    def test_empty_schema_when_get_collection_fails(self):
        """If get_collection raises, proceed with empty schema (all indexes created)."""
        from app.core.clients import qdrant as qdrant_mod
        from app.core.clients.qdrant import _PAYLOAD_INDEXES

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("collection not found")

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            qdrant_mod._ensure_payload_indexes("test_collection")

        # All indexes should be created since existing_schema is {}
        assert mock_client.create_payload_index.call_count == len(_PAYLOAD_INDEXES)


# ---------------------------------------------------------------------------
# _ensure_sparse_vector_field
# ---------------------------------------------------------------------------

class TestEnsureSparseVectorField:
    """Tests for _ensure_sparse_vector_field()."""

    def test_calls_update_collection_with_sparse_config(self):
        """Should call update_collection with the sparse vector config when imports succeed."""
        from app.core.clients import qdrant as qdrant_mod

        mock_client = Mock()

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            qdrant_mod._ensure_sparse_vector_field("my_collection")

        mock_client.update_collection.assert_called_once()
        call_kwargs = mock_client.update_collection.call_args.kwargs
        assert call_kwargs.get("collection_name") == "my_collection"
        assert "sparse_vectors_config" in call_kwargs

    def test_non_fatal_on_update_collection_exception(self):
        """Exceptions from update_collection are logged but not re-raised."""
        from app.core.clients import qdrant as qdrant_mod

        mock_client = Mock()
        mock_client.update_collection.side_effect = RuntimeError("server error")

        with patch.object(qdrant_mod, "qdrant_client", mock_client):
            # Should not raise
            qdrant_mod._ensure_sparse_vector_field("my_collection")

    def test_uses_sparse_vector_name_from_settings(self):
        """The sparse vector name should come from settings.SPARSE_VECTOR_NAME."""
        from app.core.clients import qdrant as qdrant_mod

        mock_client = Mock()

        with patch.object(qdrant_mod, "qdrant_client", mock_client), \
             patch("app.config.settings") as mock_settings:
            mock_settings.SPARSE_VECTOR_NAME = "custom_bm25"
            qdrant_mod._ensure_sparse_vector_field("col")

        # The sparse_vectors_config key should use the configured name
        call_kwargs = mock_client.update_collection.call_args.kwargs
        assert "custom_bm25" in call_kwargs.get("sparse_vectors_config", {})


# ---------------------------------------------------------------------------
# batch_points
# ---------------------------------------------------------------------------

class TestBatchPoints:
    """Tests for batch_points() generator."""

    def test_empty_list_yields_nothing(self):
        from app.core.clients.qdrant import batch_points
        result = list(batch_points([], batch_size=10))
        assert result == []

    def test_single_batch_when_points_less_than_batch_size(self):
        from app.core.clients.qdrant import batch_points
        points = list(range(5))
        result = list(batch_points(points, batch_size=10))
        assert result == [list(range(5))]

    def test_exact_multiple_batches(self):
        from app.core.clients.qdrant import batch_points
        points = list(range(6))
        result = list(batch_points(points, batch_size=2))
        assert result == [[0, 1], [2, 3], [4, 5]]

    def test_last_batch_smaller_than_batch_size(self):
        from app.core.clients.qdrant import batch_points
        points = list(range(7))
        result = list(batch_points(points, batch_size=3))
        assert result == [[0, 1, 2], [3, 4, 5], [6]]

    def test_default_batch_size_is_100(self):
        from app.core.clients.qdrant import batch_points
        points = list(range(250))
        result = list(batch_points(points))
        assert len(result) == 3  # 100 + 100 + 50

    def test_all_points_preserved(self):
        from app.core.clients.qdrant import batch_points
        points = list(range(123))
        all_items = [item for batch in batch_points(points, batch_size=10) for item in batch]
        assert all_items == points