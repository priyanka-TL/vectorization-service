"""
Unit tests for app/services/similarity_service.py

Focus on changes in this PR:
- Uses query_points(...).points instead of search()
- Exclusion via must_not (not invert=True on FieldCondition)
- Filter structure: must=[company condition], must_not=[exclude condition] or None
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


def _make_hit(source_id="src-1", score=0.9):
    hit = Mock()
    hit.id = "pid-1"
    hit.score = score
    hit.payload = {
        "source_id": source_id,
        "metadata": {"type": "pdf"},
        "text": "This is a sample text that is long enough for preview.",
    }
    return hit


class TestSimilarityServiceCheckSimilarity:
    """Tests for SimilarityService.check_similarity()."""

    def _make_service(self):
        from app.services.similarity_service import SimilarityService
        return SimilarityService()

    def _make_request(self, **kwargs):
        from app.models.api_models import SimilarityCheckRequest
        defaults = dict(text="some document text", company_id="company-abc",
                        threshold=0.85)
        defaults.update(kwargs)
        return SimilarityCheckRequest(**defaults)

    def test_returns_has_similar_true_when_results_found(self):
        svc = self._make_service()
        hit = _make_hit(source_id="src-match", score=0.92)

        mock_result = Mock()
        mock_result.points = [hit]

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            response = svc.check_similarity(self._make_request())

        assert response.has_similar is True
        assert len(response.similar_documents) == 1
        assert response.similar_documents[0]["source_id"] == "src-match"

    def test_returns_has_similar_false_when_no_results(self):
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            response = svc.check_similarity(self._make_request())

        assert response.has_similar is False
        assert response.similar_documents == []

    def test_uses_query_points_not_search(self):
        """The service must call query_points(), NOT the removed search()."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(self._make_request())

        mock_qdrant.query_points.assert_called_once()
        mock_qdrant.search.assert_not_called() if hasattr(mock_qdrant, "search") else None

    def test_query_points_called_with_named_vector(self):
        """query_points must use using='text' for named-vector collections."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(self._make_request())

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs.get("using") == "text"

    def test_exclude_source_id_goes_into_must_not(self):
        """When exclude_source_id is provided, exclusion must be in must_not, not must."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        req = self._make_request(exclude_source_id="src-exclude")

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(req)

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        search_filter = call_kwargs.get("query_filter")

        # must_not should be populated with the exclusion condition
        assert search_filter is not None
        assert search_filter.must_not is not None
        assert len(search_filter.must_not) == 1

        excluded_condition = search_filter.must_not[0]
        assert excluded_condition.match.value == "src-exclude"

        # The exclusion must NOT appear in must (no `invert` hack)
        for condition in (search_filter.must or []):
            assert condition.key != "source_id"

    def test_no_exclude_source_id_must_not_is_none(self):
        """When exclude_source_id is not provided, must_not should be None."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(self._make_request())

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        search_filter = call_kwargs.get("query_filter")
        assert search_filter.must_not is None

    def test_company_id_in_must_filter(self):
        """The company_id filter must be in must[], not must_not[]."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        req = self._make_request(company_id="target-company")

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(req)

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        search_filter = call_kwargs.get("query_filter")
        must_values = [c.match.value for c in (search_filter.must or [])]
        assert "target-company" in must_values

    def test_score_threshold_passed_to_query_points(self):
        """The threshold from the request must be forwarded to query_points."""
        svc = self._make_service()
        mock_result = Mock()
        mock_result.points = []

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        req = self._make_request(threshold=0.92)

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            svc.check_similarity(req)

        call_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert call_kwargs.get("score_threshold") == pytest.approx(0.92)

    def test_raises_http_exception_on_error(self):
        """Exceptions from qdrant should be caught and re-raised as HTTPException."""
        from fastapi import HTTPException
        svc = self._make_service()

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.side_effect = RuntimeError("server error")
            with pytest.raises(HTTPException) as exc_info:
                svc.check_similarity(self._make_request())

        assert exc_info.value.status_code == 500

    def test_result_shape_matches_expected_fields(self):
        """Each similar_document must have source_id, similarity_score, metadata, text_preview, chunk_id."""
        svc = self._make_service()
        hit = _make_hit(source_id="src-abc", score=0.87)

        mock_result = Mock()
        mock_result.points = [hit]

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.0] * 384

        with patch("app.services.similarity_service.generate_single_embedding",
                   return_value=mock_embedding), \
             patch("app.services.similarity_service.qdrant_client") as mock_qdrant:
            mock_qdrant.query_points.return_value = mock_result
            response = svc.check_similarity(self._make_request())

        doc = response.similar_documents[0]
        assert "source_id" in doc
        assert "similarity_score" in doc
        assert "metadata" in doc
        assert "text_preview" in doc
        assert "chunk_id" in doc
        assert doc["similarity_score"] == pytest.approx(0.87)