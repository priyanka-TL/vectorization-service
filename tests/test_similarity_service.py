"""
Unit tests for app/services/similarity_service.py

Tests focus on the changes introduced in this PR:
- Migration from qdrant_client.search() to qdrant_client.query_points()
- Correct use of 'using="text"' named vector
- exclude_source_id uses must_not clause (not FieldCondition.invert)
- Results read from query_points().points
- No exclude_source_id → must_not is None/empty
- Exception → HTTPException 500
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_query_points_response(hits):
    """Build a mock object that mimics query_points() return value."""
    response = Mock()
    response.points = hits
    return response


def _make_hit(source_id="src-1", score=0.9, text="sample text", metadata=None):
    hit = Mock()
    hit.id = "chunk-id-1"
    hit.score = score
    hit.payload = {
        "source_id": source_id,
        "text": text,
        "metadata": metadata or {"type": "pdf"},
    }
    return hit


def _make_embedding():
    emb = Mock()
    emb.tolist.return_value = [0.1] * 384
    return emb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSimilarityServiceCheckSimilarity:
    """Tests for SimilarityService.check_similarity()."""

    def _make_request(self, text="some text", company_id="co-1",
                      threshold=0.85, exclude_source_id=None):
        from app.models.api_models import SimilarityCheckRequest
        return SimilarityCheckRequest(
            text=text,
            company_id=company_id,
            threshold=threshold,
            exclude_source_id=exclude_source_id,
        )

    def test_calls_query_points_not_search(self):
        """query_points must be called; the old search() must not be called."""
        mock_qp_response = _make_query_points_response([])
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = mock_qp_response

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            svc.check_similarity(self._make_request())

        mock_qdrant.query_points.assert_called_once()
        mock_qdrant.search.assert_not_called()

    def test_uses_text_named_vector(self):
        """The call must pass using='text'."""
        mock_qp_response = _make_query_points_response([])
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = mock_qp_response

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            svc.check_similarity(self._make_request())

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs.get("using") == "text"

    def test_no_similar_documents_found(self):
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            result = svc.check_similarity(self._make_request())

        assert result.has_similar is False
        assert result.similar_documents == []

    def test_similar_documents_returned(self):
        hit = _make_hit(source_id="src-99", score=0.92)
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([hit])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            result = svc.check_similarity(self._make_request())

        assert result.has_similar is True
        assert len(result.similar_documents) == 1
        assert result.similar_documents[0]["source_id"] == "src-99"
        assert result.similar_documents[0]["similarity_score"] == pytest.approx(0.92)

    def test_exclude_source_id_uses_must_not(self):
        """When exclude_source_id is set, the filter must have a must_not clause."""
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            req = self._make_request(exclude_source_id="excluded-src")
            svc.check_similarity(req)

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        assert query_filter is not None
        # must_not should be set (non-empty)
        assert query_filter.must_not is not None
        assert len(query_filter.must_not) > 0

        # Check the excluded source_id is in the must_not condition
        must_not_field = query_filter.must_not[0]
        assert must_not_field.key == "source_id"
        assert must_not_field.match.value == "excluded-src"

    def test_no_exclude_source_id_must_not_is_none(self):
        """When no exclude_source_id, must_not should be None."""
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            svc.check_similarity(self._make_request(exclude_source_id=None))

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        # must_not should be None when no exclusion
        assert query_filter.must_not is None

    def test_results_read_from_points_attribute(self):
        """query_points().points must be iterated, not the response itself."""
        hit = _make_hit()
        response = Mock()
        response.points = [hit]
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = response

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            result = svc.check_similarity(self._make_request())

        assert result.has_similar is True

    def test_exception_raises_http_500(self):
        """Any exception should be wrapped in HTTPException 500."""
        from fastapi import HTTPException

        mock_qdrant = Mock()
        mock_qdrant.query_points.side_effect = RuntimeError("connection lost")

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            with pytest.raises(HTTPException) as exc_info:
                svc.check_similarity(self._make_request())

        assert exc_info.value.status_code == 500

    def test_text_preview_truncated_to_200_chars(self):
        """text_preview in result should be limited to 200 chars + '...'."""
        long_text = "x" * 500
        hit = _make_hit(text=long_text)
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([hit])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            result = svc.check_similarity(self._make_request())

        preview = result.similar_documents[0]["text_preview"]
        # 200 chars of "x" + "..."
        assert preview == "x" * 200 + "..."

    def test_company_filter_applied(self):
        """must clause should filter by company_id."""
        mock_qdrant = Mock()
        mock_qdrant.query_points.return_value = _make_query_points_response([])

        with patch("app.services.similarity_service.qdrant_client", mock_qdrant), \
             patch("app.services.similarity_service.generate_single_embedding",
                   return_value=_make_embedding()):
            from app.services.similarity_service import SimilarityService
            svc = SimilarityService()
            svc.check_similarity(self._make_request(company_id="my-company"))

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        query_filter = call_kwargs.get("query_filter")
        # must clause should have a company filter
        must_conditions = query_filter.must
        company_condition = next(
            (c for c in must_conditions if c.key == "metadata.company"), None
        )
        assert company_condition is not None
        assert company_condition.match.value == "my-company"