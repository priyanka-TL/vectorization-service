"""
Unit tests for app/services/text_embedding_search_service.py

Tests cover the PR changes:
- query_points() is called (not the removed search())
- using="text" named vector
- Results read from .points attribute
- Threshold filtering applied client-side
- Sources without source_id are skipped
- Results are capped at top_k
- Empty result set returns an empty response
"""
import os
import pytest
from unittest.mock import Mock, patch

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding():
    emb = Mock()
    emb.tolist.return_value = [0.1] * 384
    return emb


def _make_point(source_id="src-1", score=0.9, text="chunk text", metadata=None):
    p = Mock()
    p.score = score
    p.payload = {
        "source_id": source_id,
        "text": text,
        "metadata": metadata or {},
    }
    return p


def _make_qp_response(points):
    r = Mock()
    r.points = points
    return r


def _make_request(query="find info", top_k=10, threshold=0.5):
    from app.models.api_models import TextSearchRequest
    return TextSearchRequest(query=query, top_k=top_k, threshold=threshold)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextEmbeddingSearchServiceSearch:

    def test_calls_query_points_not_search(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            svc.search(_make_request())

        mock_qdrant.query_points.assert_called_once()
        mock_qdrant.search.assert_not_called()

    def test_uses_text_named_vector(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            svc.search(_make_request())

        kwargs = mock_qdrant.query_points.call_args.kwargs
        assert kwargs.get("using") == "text"

    def test_with_payload_true_is_passed(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            svc.search(_make_request())

        kwargs = mock_qdrant.query_points.call_args.kwargs
        assert kwargs.get("with_payload") is True

    def test_empty_results_returns_empty_response(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request())

        assert result.total_results == 0
        assert result.results == []

    def test_results_above_threshold_included(self):
        points = [
            _make_point(source_id="src-1", score=0.9),
            _make_point(source_id="src-2", score=0.8),
        ]
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response(points)

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(threshold=0.7))

        assert result.total_results == 2

    def test_results_below_threshold_excluded(self):
        points = [
            _make_point(source_id="src-1", score=0.9),
            _make_point(source_id="src-2", score=0.4),  # below threshold=0.5
        ]
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response(points)

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(threshold=0.5))

        assert result.total_results == 1
        assert result.results[0].source_id == "src-1"

    def test_points_without_source_id_skipped(self):
        points = [
            _make_point(source_id=None, score=0.9),  # no source_id
            _make_point(source_id="", score=0.85),   # empty string
            _make_point(source_id="src-3", score=0.8),
        ]
        # Make the None source_id point return None from payload.get
        points[0].payload = {"text": "text", "score": 0.9, "metadata": {}}
        points[1].payload = {"source_id": "", "text": "text", "metadata": {}}

        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response(points)

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(threshold=0.0))

        # Only src-3 should be included
        source_ids = [r.source_id for r in result.results]
        assert "src-3" in source_ids
        assert len([s for s in source_ids if s]) == 1

    def test_results_capped_at_top_k(self):
        points = [_make_point(source_id=f"src-{i}", score=0.9 - i * 0.01) for i in range(20)]
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response(points)

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(top_k=5, threshold=0.0))

        assert result.total_results <= 5

    def test_response_includes_query_text(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(query="specific query"))

        assert result.query == "specific query"

    def test_result_items_have_correct_fields(self):
        point = _make_point(source_id="src-1", score=0.88, text="doc text",
                            metadata={"type": "pdf"})
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([point])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            result = svc.search(_make_request(threshold=0.0))

        item = result.results[0]
        assert item.source_id == "src-1"
        assert item.text == "doc text"
        assert item.score == pytest.approx(0.88)
        assert item.metadata == {"type": "pdf"}

    def test_exception_propagates(self):
        """Exceptions should not be swallowed."""
        mock_qdrant = Mock()
        mock_qdrant.query_points.side_effect = RuntimeError("qdrant error")

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            with pytest.raises(RuntimeError):
                svc.search(_make_request())

    def test_limit_passed_to_query_points(self):
        """top_k must be forwarded as limit to query_points."""
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_qp_response([])

        with patch("app.services.text_embedding_search_service.qdrant_client", mock_qdrant), \
             patch("app.services.text_embedding_search_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.text_embedding_search_service import TextEmbeddingSearchService
            svc = TextEmbeddingSearchService()
            svc.search(_make_request(top_k=42))

        kwargs = mock_qdrant.query_points.call_args.kwargs
        assert kwargs.get("limit") == 42