"""
Unit tests for app/models/api_models.py

Tests cover only the fields/changes introduced in this PR:
- PrioritizedSearchRequest.search_mode (new field, default "hybrid")
- SearchResultItem.keyword_score (new optional field)
- SearchResultItem.title_match (new optional field)
- SearchResultItem.summary_match (new optional field)
"""
import os
import pytest
from typing import Optional

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# PrioritizedSearchRequest — search_mode field
# ---------------------------------------------------------------------------

class TestPrioritizedSearchRequestSearchMode:
    """Tests for the new search_mode field on PrioritizedSearchRequest."""

    def test_default_search_mode_is_hybrid(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(query="test", top_k=5)
        assert req.search_mode == "hybrid"

    def test_search_mode_semantic(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(query="test", top_k=5, search_mode="semantic")
        assert req.search_mode == "semantic"

    def test_search_mode_keyword(self):
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(query="test", top_k=5, search_mode="keyword")
        assert req.search_mode == "keyword"

    def test_search_mode_custom_string(self):
        """Custom strings should be accepted (no enum validation enforced)."""
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(query="test", top_k=5, search_mode="dense_only")
        assert req.search_mode == "dense_only"

    def test_search_mode_present_when_serialised(self):
        """search_mode should appear in the model dict."""
        from app.models.api_models import PrioritizedSearchRequest
        req = PrioritizedSearchRequest(query="test", top_k=5)
        data = req.model_dump()
        assert "search_mode" in data
        assert data["search_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# SearchResultItem — new optional fields
# ---------------------------------------------------------------------------

class TestSearchResultItemNewFields:
    """Tests for keyword_score, title_match, and summary_match on SearchResultItem."""

    def _base_item(self, **kwargs):
        """Helper: create a SearchResultItem with required fields populated."""
        from app.models.api_models import SearchResultItem
        defaults = dict(
            id="test-id",
            text="some text",
            score=0.8,
            source_id="src-1",
            metadata={},
        )
        defaults.update(kwargs)
        return SearchResultItem(**defaults)

    # keyword_score ──────────────────────────────────────────────────────────

    def test_keyword_score_defaults_to_none(self):
        item = self._base_item()
        assert item.keyword_score is None

    def test_keyword_score_accepts_float(self):
        item = self._base_item(keyword_score=0.75)
        assert item.keyword_score == pytest.approx(0.75)

    def test_keyword_score_accepts_zero(self):
        item = self._base_item(keyword_score=0.0)
        assert item.keyword_score == 0.0

    def test_keyword_score_accepts_one(self):
        item = self._base_item(keyword_score=1.0)
        assert item.keyword_score == 1.0

    # title_match ────────────────────────────────────────────────────────────

    def test_title_match_defaults_to_none(self):
        item = self._base_item()
        assert item.title_match is None

    def test_title_match_exact(self):
        item = self._base_item(title_match="exact")
        assert item.title_match == "exact"

    def test_title_match_partial(self):
        item = self._base_item(title_match="partial")
        assert item.title_match == "partial"

    # summary_match ──────────────────────────────────────────────────────────

    def test_summary_match_defaults_to_none(self):
        item = self._base_item()
        assert item.summary_match is None

    def test_summary_match_exact(self):
        item = self._base_item(summary_match="exact")
        assert item.summary_match == "exact"

    def test_summary_match_partial(self):
        item = self._base_item(summary_match="partial")
        assert item.summary_match == "partial"

    # All new fields present in serialised output ────────────────────────────

    def test_new_fields_present_in_model_dump(self):
        item = self._base_item(keyword_score=0.6, title_match="partial", summary_match="exact")
        data = item.model_dump()
        assert "keyword_score" in data
        assert "title_match" in data
        assert "summary_match" in data
        assert data["keyword_score"] == pytest.approx(0.6)
        assert data["title_match"] == "partial"
        assert data["summary_match"] == "exact"

    def test_all_new_fields_none_when_not_provided(self):
        item = self._base_item()
        data = item.model_dump()
        assert data["keyword_score"] is None
        assert data["title_match"] is None
        assert data["summary_match"] is None

    def test_item_with_all_fields_populated(self):
        """Full smoke-test: construct with every new field."""
        from app.models.api_models import SearchResultItem
        item = SearchResultItem(
            id="abc",
            text="document text",
            score=0.9,
            source_id="source-42",
            metadata={"type": "pdf"},
            keyword_score=0.55,
            title_match="exact",
            summary_match="partial",
        )
        assert item.keyword_score == pytest.approx(0.55)
        assert item.title_match == "exact"
        assert item.summary_match == "partial"