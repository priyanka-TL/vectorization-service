"""
Unit tests for app/models/api_models.py

Focus on the new fields added in this PR:
- PrioritizedSearchRequest.search_mode (default "hybrid", accepted values)
- SearchResultItem.keyword_score (Optional[float], default None)
- SearchResultItem.title_match (Optional[str], default None)
- SearchResultItem.summary_match (Optional[str], default None)
"""
import os
import pytest
from pydantic import ValidationError

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


class TestPrioritizedSearchRequestSearchMode:
    """Tests for the new search_mode field on PrioritizedSearchRequest."""

    def test_default_search_mode_is_hybrid(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest()
        assert req.search_mode == "hybrid"

    def test_search_mode_semantic_accepted(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(search_mode="semantic")
        assert req.search_mode == "semantic"

    def test_search_mode_keyword_accepted(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(search_mode="keyword")
        assert req.search_mode == "keyword"

    def test_search_mode_hybrid_explicitly_set(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(search_mode="hybrid")
        assert req.search_mode == "hybrid"

    def test_search_mode_any_string_accepted(self):
        """The field is typed as str with no enum constraint — any string is valid."""
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(search_mode="custom_mode")
        assert req.search_mode == "custom_mode"

    def test_other_fields_still_have_defaults(self):
        """Adding search_mode must not break existing field defaults."""
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest()
        assert req.query is None
        assert req.categories is None
        assert req.organizations is None
        assert req.resource_type is None
        assert req.file_type is None


class TestSearchResultItemNewFields:
    """Tests for new optional fields on SearchResultItem."""

    def _make_result_item(self, **kwargs):
        from app.models.api_models import SearchResultItem
        defaults = dict(
            id="test-id",
            text="sample text",
            metadata={},
            source_id="src-1",
            score=0.8,
        )
        defaults.update(kwargs)
        return SearchResultItem(**defaults)

    def test_keyword_score_defaults_to_none(self):
        item = self._make_result_item()
        assert item.keyword_score is None

    def test_title_match_defaults_to_none(self):
        item = self._make_result_item()
        assert item.title_match is None

    def test_summary_match_defaults_to_none(self):
        item = self._make_result_item()
        assert item.summary_match is None

    def test_keyword_score_accepts_float(self):
        item = self._make_result_item(keyword_score=0.42)
        assert item.keyword_score == pytest.approx(0.42)

    def test_title_match_exact(self):
        item = self._make_result_item(title_match="exact")
        assert item.title_match == "exact"

    def test_title_match_partial(self):
        item = self._make_result_item(title_match="partial")
        assert item.title_match == "partial"

    def test_summary_match_exact(self):
        item = self._make_result_item(summary_match="exact")
        assert item.summary_match == "exact"

    def test_summary_match_partial(self):
        item = self._make_result_item(summary_match="partial")
        assert item.summary_match == "partial"

    def test_all_new_fields_set_together(self):
        item = self._make_result_item(
            keyword_score=0.75,
            title_match="partial",
            summary_match="exact",
        )
        assert item.keyword_score == pytest.approx(0.75)
        assert item.title_match == "partial"
        assert item.summary_match == "exact"

    def test_field_scores_stays_separate_from_match_types(self):
        """field_scores should hold only numeric scores, not match-type strings."""
        item = self._make_result_item(
            field_scores={"title": 0.9, "text": 0.7},
            title_match="exact",
        )
        assert "title_match" not in item.field_scores
        assert item.title_match == "exact"

    def test_serialization_includes_new_fields(self):
        """model_dump() must include all three new optional fields."""
        item = self._make_result_item(
            keyword_score=0.3,
            title_match="partial",
            summary_match=None,
        )
        d = item.model_dump()
        assert "keyword_score" in d
        assert "title_match" in d
        assert "summary_match" in d
        assert d["keyword_score"] == pytest.approx(0.3)
        assert d["title_match"] == "partial"
        assert d["summary_match"] is None

    def test_none_keyword_score_serialises_as_null(self):
        item = self._make_result_item()
        d = item.model_dump()
        assert d["keyword_score"] is None