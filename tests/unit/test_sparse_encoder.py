"""
Unit tests for app/core/clients/sparse_encoder.py

Tests cover:
- generate_sparse_vector: empty text handling, successful encoding, error propagation
- is_sparse_available: availability check with mocked encoder
- _get_sparse_encoder: singleton caching behaviour
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestGenerateSparseVector:
    """Tests for generate_sparse_vector()"""

    def test_empty_string_returns_empty_lists(self):
        """Empty string input should return ([], []) without calling the encoder."""
        import app.core.clients.sparse_encoder as mod
        indices, values = mod.generate_sparse_vector("")
        assert indices == []
        assert values == []

    def test_whitespace_only_returns_empty_lists(self):
        """Whitespace-only input should return ([], []) without calling the encoder."""
        import app.core.clients.sparse_encoder as mod
        indices, values = mod.generate_sparse_vector("   \t\n  ")
        assert indices == []
        assert values == []

    def test_successful_encoding_returns_parallel_lists(self):
        """Successful BM25 encoding should return matching indices and values lists."""
        import app.core.clients.sparse_encoder as mod

        # Build a fake embedding result with numpy-style arrays
        fake_embedding = Mock()
        fake_embedding.indices.tolist.return_value = [1, 5, 42]
        fake_embedding.values.tolist.return_value = [0.1, 0.5, 0.9]

        fake_encoder = Mock()
        fake_encoder.embed.return_value = iter([fake_embedding])

        # Reset singleton so we control it
        original = mod._sparse_encoder
        mod._sparse_encoder = fake_encoder
        try:
            indices, values = mod.generate_sparse_vector("insurance document")
            assert indices == [1, 5, 42]
            assert values == [0.1, 0.5, 0.9]
            fake_encoder.embed.assert_called_once_with(["insurance document"])
        finally:
            mod._sparse_encoder = original

    def test_encoder_returning_empty_results_gives_empty_lists(self):
        """If the encoder returns an empty iterator, return ([], [])."""
        import app.core.clients.sparse_encoder as mod

        fake_encoder = Mock()
        fake_encoder.embed.return_value = iter([])  # no embeddings

        original = mod._sparse_encoder
        mod._sparse_encoder = fake_encoder
        try:
            indices, values = mod.generate_sparse_vector("some text")
            assert indices == []
            assert values == []
        finally:
            mod._sparse_encoder = original

    def test_encoder_exception_raises_runtime_error(self):
        """If the encoder raises, generate_sparse_vector should raise RuntimeError."""
        import app.core.clients.sparse_encoder as mod

        fake_encoder = Mock()
        fake_encoder.embed.side_effect = ValueError("encoding error")

        original = mod._sparse_encoder
        mod._sparse_encoder = fake_encoder
        try:
            with pytest.raises(RuntimeError, match="Sparse vector generation failed"):
                mod.generate_sparse_vector("some text")
        finally:
            mod._sparse_encoder = original

    def test_none_text_returns_empty_lists(self):
        """None is treated as falsy; should return ([], []) without error."""
        import app.core.clients.sparse_encoder as mod
        # generate_sparse_vector checks `not text` before anything else
        indices, values = mod.generate_sparse_vector(None)  # type: ignore[arg-type]
        assert indices == []
        assert values == []

    def test_returns_tuple_of_two_lists(self):
        """Return type must always be a 2-tuple of lists."""
        import app.core.clients.sparse_encoder as mod

        fake_embedding = Mock()
        fake_embedding.indices.tolist.return_value = [0, 7]
        fake_embedding.values.tolist.return_value = [0.3, 0.7]

        fake_encoder = Mock()
        fake_encoder.embed.return_value = iter([fake_embedding])

        original = mod._sparse_encoder
        mod._sparse_encoder = fake_encoder
        try:
            result = mod.generate_sparse_vector("hello world")
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], list)
            assert isinstance(result[1], list)
        finally:
            mod._sparse_encoder = original


class TestIsSparseAvailable:
    """Tests for is_sparse_available()"""

    def test_returns_true_when_encoder_loads_successfully(self):
        """Should return True when the encoder singleton can be obtained."""
        import app.core.clients.sparse_encoder as mod

        original = mod._sparse_encoder
        mod._sparse_encoder = Mock()  # already loaded — _get_sparse_encoder returns it
        try:
            result = mod.is_sparse_available()
            assert result is True
        finally:
            mod._sparse_encoder = original

    def test_returns_false_when_encoder_raises(self):
        """Should return False when the encoder cannot be initialised."""
        import app.core.clients.sparse_encoder as mod

        original = mod._sparse_encoder
        mod._sparse_encoder = None  # force re-initialisation
        try:
            # Patch fastembed import to raise so _get_sparse_encoder fails
            with patch.dict("sys.modules", {"fastembed": None}):
                result = mod.is_sparse_available()
                # fastembed not importable → should return False, not raise
                assert result is False
        finally:
            mod._sparse_encoder = original


class TestGetSparseEncoderSingleton:
    """Tests for _get_sparse_encoder() singleton caching."""

    def test_cached_encoder_is_returned_without_reinitialising(self):
        """If the singleton is already set, _get_sparse_encoder must return it without creating a new one."""
        import app.core.clients.sparse_encoder as mod

        sentinel = object()
        original = mod._sparse_encoder
        mod._sparse_encoder = sentinel
        try:
            result = mod._get_sparse_encoder()
            assert result is sentinel
        finally:
            mod._sparse_encoder = original

    def test_raises_when_fastembed_not_importable(self):
        """When fastembed is unavailable, _get_sparse_encoder should propagate the exception."""
        import app.core.clients.sparse_encoder as mod

        original = mod._sparse_encoder
        mod._sparse_encoder = None
        try:
            with patch.dict("sys.modules", {"fastembed": None}):
                with pytest.raises(Exception):
                    mod._get_sparse_encoder()
        finally:
            mod._sparse_encoder = original