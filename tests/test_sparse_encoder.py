"""
Unit tests for app/core/clients/sparse_encoder.py

Tests cover:
- Lazy singleton initialization
- generate_sparse_vector() with valid text, empty text, whitespace-only text
- generate_sparse_vector() raises RuntimeError when encoder fails
- is_sparse_available() returns True when encoder loads successfully
- is_sparse_available() returns False when fastembed is unavailable
- Caching behaviour (encoder loaded only once)
"""
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_sparse_encoder_singleton():
    """Reset the module-level singleton before every test."""
    import app.core.clients.sparse_encoder as mod
    original = mod._sparse_encoder
    mod._sparse_encoder = None
    yield
    mod._sparse_encoder = original


def _make_mock_embedding(indices, values):
    """Return a mock SparseEmbedding with numpy-like .tolist() support."""
    emb = Mock()
    emb.indices = Mock()
    emb.indices.tolist.return_value = indices
    emb.values = Mock()
    emb.values.tolist.return_value = values
    return emb


# ---------------------------------------------------------------------------
# _get_sparse_encoder
# ---------------------------------------------------------------------------

class TestGetSparseEncoder:
    """Tests for the lazy-loading singleton _get_sparse_encoder()."""

    def test_loads_fastembed_on_first_call(self):
        """_get_sparse_encoder loads SparseTextEmbedding when called the first time."""
        mock_model_instance = Mock()
        mock_ste_cls = Mock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"fastembed": MagicMock(SparseTextEmbedding=mock_ste_cls)}):
            from app.core.clients.sparse_encoder import _get_sparse_encoder, _SPARSE_MODEL
            result = _get_sparse_encoder()

        mock_ste_cls.assert_called_once_with(model_name=_SPARSE_MODEL)
        assert result is mock_model_instance

    def test_caches_encoder_across_calls(self):
        """_get_sparse_encoder returns the same object on repeated calls."""
        mock_model_instance = Mock()
        mock_ste_cls = Mock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"fastembed": MagicMock(SparseTextEmbedding=mock_ste_cls)}):
            from app.core.clients.sparse_encoder import _get_sparse_encoder
            first = _get_sparse_encoder()
            second = _get_sparse_encoder()

        assert first is second
        # SparseTextEmbedding constructor should only have been called once
        assert mock_ste_cls.call_count == 1

    def test_raises_when_fastembed_missing(self):
        """_get_sparse_encoder raises when fastembed cannot be imported."""
        with patch.dict(sys.modules, {"fastembed": None}):
            from app.core.clients.sparse_encoder import _get_sparse_encoder
            with pytest.raises(Exception):
                _get_sparse_encoder()

    def test_raises_when_model_init_fails(self):
        """_get_sparse_encoder raises when the model constructor throws."""
        mock_ste_cls = Mock(side_effect=RuntimeError("model download failed"))
        with patch.dict(sys.modules, {"fastembed": MagicMock(SparseTextEmbedding=mock_ste_cls)}):
            from app.core.clients.sparse_encoder import _get_sparse_encoder
            with pytest.raises(Exception):
                _get_sparse_encoder()


# ---------------------------------------------------------------------------
# generate_sparse_vector
# ---------------------------------------------------------------------------

class TestGenerateSparseVector:
    """Tests for generate_sparse_vector()."""

    def test_returns_empty_lists_for_empty_string(self):
        from app.core.clients.sparse_encoder import generate_sparse_vector
        indices, values = generate_sparse_vector("")
        assert indices == []
        assert values == []

    def test_returns_empty_lists_for_whitespace_only(self):
        from app.core.clients.sparse_encoder import generate_sparse_vector
        indices, values = generate_sparse_vector("   \t\n  ")
        assert indices == []
        assert values == []

    def test_returns_indices_and_values_for_valid_text(self):
        """Should return two parallel lists for normal text."""
        mock_emb = _make_mock_embedding([1, 2, 3], [0.5, 0.3, 0.2])
        mock_encoder = Mock()
        mock_encoder.embed.return_value = [mock_emb]

        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            indices, values = generate_sparse_vector("hello world")

        assert indices == [1, 2, 3]
        assert values == [0.5, 0.3, 0.2]
        mock_encoder.embed.assert_called_once_with(["hello world"])

    def test_returns_empty_when_embed_returns_no_results(self):
        """Edge case: encoder returns an empty result list."""
        mock_encoder = Mock()
        mock_encoder.embed.return_value = []

        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            indices, values = generate_sparse_vector("some text")

        assert indices == []
        assert values == []

    def test_raises_runtime_error_on_encoder_failure(self):
        """Should propagate a RuntimeError when the encoder raises."""
        mock_encoder = Mock()
        mock_encoder.embed.side_effect = ValueError("encoding failed")

        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            with pytest.raises(RuntimeError, match="Sparse vector generation failed"):
                generate_sparse_vector("test text")

    def test_raises_runtime_error_when_encoder_unavailable(self):
        """Should raise RuntimeError when _get_sparse_encoder itself raises."""
        with patch("app.core.clients.sparse_encoder._get_sparse_encoder",
                   side_effect=ImportError("fastembed not installed")):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            with pytest.raises(RuntimeError, match="Sparse vector generation failed"):
                generate_sparse_vector("test text")

    def test_indices_and_values_are_lists(self):
        """Return type must be (list[int], list[float]), not numpy arrays."""
        mock_emb = _make_mock_embedding([10, 20], [1.0, 0.5])
        mock_encoder = Mock()
        mock_encoder.embed.return_value = [mock_emb]

        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            indices, values = generate_sparse_vector("text")

        assert isinstance(indices, list)
        assert isinstance(values, list)

    def test_single_token_text(self):
        """Single-word input should still be encoded."""
        mock_emb = _make_mock_embedding([5], [1.0])
        mock_encoder = Mock()
        mock_encoder.embed.return_value = [mock_emb]

        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import generate_sparse_vector
            indices, values = generate_sparse_vector("hello")

        assert indices == [5]
        assert values == [1.0]


# ---------------------------------------------------------------------------
# is_sparse_available
# ---------------------------------------------------------------------------

class TestIsSparseAvailable:
    """Tests for is_sparse_available()."""

    def test_returns_true_when_encoder_loads(self):
        mock_encoder = Mock()
        with patch("app.core.clients.sparse_encoder._get_sparse_encoder", return_value=mock_encoder):
            from app.core.clients.sparse_encoder import is_sparse_available
            assert is_sparse_available() is True

    def test_returns_false_when_encoder_raises(self):
        with patch("app.core.clients.sparse_encoder._get_sparse_encoder",
                   side_effect=ImportError("no fastembed")):
            from app.core.clients.sparse_encoder import is_sparse_available
            assert is_sparse_available() is False

    def test_returns_false_when_model_init_fails(self):
        with patch("app.core.clients.sparse_encoder._get_sparse_encoder",
                   side_effect=RuntimeError("model error")):
            from app.core.clients.sparse_encoder import is_sparse_available
            assert is_sparse_available() is False