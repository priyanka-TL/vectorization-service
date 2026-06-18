"""
Unit tests for app/services/prioritized_search_service.py

Focus on methods added/changed in this PR:
- _classify_text_match: exact / partial / None classification
- _get_field_match_sources: scroll-based match collection
- _supplement_matches_from_results: in-memory substring scan
- _apply_field_boost: score boosting and re-sort
- _fetch_field_match_docs: floor-score injection
- _build_result_items: title_match / summary_match extraction from field_scores
- _rank_results: RRF path (new) vs weighted-sum path
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service():
    """Return a PrioritizedSearchService with mocked qdrant_client."""
    from app.services import prioritized_search_service as mod
    svc = mod.PrioritizedSearchService.__new__(mod.PrioritizedSearchService)
    svc.collection_name = "test_collection"
    svc.default_top_k = 10
    svc.max_top_k = 100
    svc.min_filter_score = 0
    svc.priority_order = ["title", "text", "tags", "summary", "metadata"]
    svc.default_weights = {
        "title": 0.36,
        "text": 0.27,
        "tags": 0.14,
        "summary": 0.14,
        "metadata": 0.09,
    }
    svc.min_score_threshold = 0.0
    return svc


def _make_point(source_id="src-1", field_value="Insurance Policy Overview",
                point_id="pid-1", field="title"):
    """Build a mock Qdrant scroll point."""
    point = Mock()
    point.id = point_id
    point.payload = {
        "source_id": source_id,
        field: field_value,
        "text": "some chunk text",
        "metadata": {},
    }
    return point


def _make_result(source_id="src-1", score=0.5, field="title",
                 field_value="Insurance Policy"):
    return {
        "id": "pid-1",
        "payload": {
            "source_id": source_id,
            field: field_value,
            "text": "chunk",
            "metadata": {},
            "title": field_value if field == "title" else None,
            "summary": field_value if field == "summary" else None,
        },
        "weighted_score": score,
        "field_scores": {},
    }


# ---------------------------------------------------------------------------
# _classify_text_match
# ---------------------------------------------------------------------------

class TestClassifyTextMatch:
    def test_exact_match(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("insurance", "insurance") == "exact"

    def test_partial_prefix_match(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("insur", "insurance policy") == "partial"

    def test_partial_infix_match(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("sur", "insurance") == "partial"

    def test_no_match_returns_none(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("xyz", "insurance policy") is None

    def test_empty_field_returns_none(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("something", "") is None

    def test_query_longer_than_field_no_match(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("insurance policy document", "insurance") is None

    def test_case_sensitive_match(self):
        """_classify_text_match receives already-lowercased strings; uppercase mismatch → None."""
        from app.services.prioritized_search_service import PrioritizedSearchService
        # Callers lower() before passing; uppercase query won't match lowercased field
        assert PrioritizedSearchService._classify_text_match("Insurance", "insurance policy") is None

    def test_exact_match_with_spaces(self):
        from app.services.prioritized_search_service import PrioritizedSearchService
        assert PrioritizedSearchService._classify_text_match("policy review", "policy review") == "exact"


# ---------------------------------------------------------------------------
# _supplement_matches_from_results
# ---------------------------------------------------------------------------

class TestSupplementMatchesFromResults:
    def test_adds_infix_match_not_in_matches(self):
        svc = _make_service()
        result = _make_result(source_id="src-2", field="title",
                              field_value="insurance coverage")
        result["payload"]["title"] = "insurance coverage"
        matches = {}
        svc._supplement_matches_from_results("sur", [result], "title", matches)
        # "sur" is infix of "insurance coverage" → partial match added
        assert "src-2" in matches
        assert matches["src-2"] == "partial"

    def test_does_not_overwrite_existing_match(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", field="title",
                              field_value="insurance")
        result["payload"]["title"] = "insurance"
        matches = {"src-1": "exact"}  # already exact
        svc._supplement_matches_from_results("insurance", [result], "title", matches)
        # Must not change existing entry
        assert matches["src-1"] == "exact"

    def test_skips_results_without_source_id(self):
        svc = _make_service()
        result = _make_result(source_id=None, field="title",
                              field_value="insurance")
        result["payload"]["source_id"] = None
        result["payload"]["title"] = "insurance"
        matches = {}
        svc._supplement_matches_from_results("insurance", [result], "title", matches)
        assert matches == {}

    def test_empty_query_does_nothing(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", field="title",
                              field_value="insurance")
        result["payload"]["title"] = "insurance"
        matches = {}
        svc._supplement_matches_from_results("   ", [result], "title", matches)
        assert matches == {}

    def test_no_match_when_query_not_in_field(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", field="title",
                              field_value="car policy")
        result["payload"]["title"] = "car policy"
        matches = {}
        svc._supplement_matches_from_results("insurance", [result], "title", matches)
        assert matches == {}


# ---------------------------------------------------------------------------
# _apply_field_boost
# ---------------------------------------------------------------------------

class TestApplyFieldBoost:
    def test_exact_match_applies_exact_boost(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", score=0.4)
        matches = {"src-1": "exact"}
        boosted = svc._apply_field_boost([result], matches, "title",
                                         exact_boost=2.5, partial_boost=1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(min(0.4 * 2.5, 1.0))
        assert boosted[0]["field_scores"]["title_match"] == "exact"

    def test_partial_match_applies_partial_boost(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", score=0.4)
        matches = {"src-1": "partial"}
        boosted = svc._apply_field_boost([result], matches, "title",
                                         exact_boost=2.5, partial_boost=1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(min(0.4 * 1.5, 1.0))
        assert boosted[0]["field_scores"]["title_match"] == "partial"

    def test_score_capped_at_one(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", score=0.9)
        matches = {"src-1": "exact"}
        boosted = svc._apply_field_boost([result], matches, "title",
                                         exact_boost=2.5, partial_boost=1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(1.0)

    def test_no_match_sets_title_match_to_none(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", score=0.4)
        matches = {}  # no matches
        boosted = svc._apply_field_boost([result], matches, "title",
                                         exact_boost=2.5, partial_boost=1.5)
        # Score unchanged
        assert boosted[0]["weighted_score"] == pytest.approx(0.4)
        assert boosted[0]["field_scores"].get("title_match") is None

    def test_results_sorted_descending_after_boost(self):
        svc = _make_service()
        r1 = _make_result(source_id="src-1", score=0.3)
        r2 = _make_result(source_id="src-2", score=0.5)
        matches = {"src-1": "exact"}  # src-1 gets boosted: 0.3*2.5=0.75
        boosted = svc._apply_field_boost([r1, r2], matches, "title",
                                         exact_boost=2.5, partial_boost=1.5)
        assert boosted[0]["payload"]["source_id"] == "src-1"  # highest after boost
        assert boosted[1]["payload"]["source_id"] == "src-2"

    def test_summary_match_key_used_for_summary_field(self):
        svc = _make_service()
        result = _make_result(source_id="src-1", score=0.4)
        matches = {"src-1": "partial"}
        boosted = svc._apply_field_boost([result], matches, "summary",
                                         exact_boost=1.4, partial_boost=1.2)
        assert "summary_match" in boosted[0]["field_scores"]
        assert boosted[0]["field_scores"]["summary_match"] == "partial"


# ---------------------------------------------------------------------------
# _fetch_field_match_docs
# ---------------------------------------------------------------------------

class TestFetchFieldMatchDocs:
    def test_returns_injected_doc_with_floor_score(self):
        svc = _make_service()
        point = _make_point(source_id="src-missing", field_value="Insurance Policy")
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([point], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            injected = svc._fetch_field_match_docs(
                ["src-missing"], {"src-missing": "exact"}, "title",
                exact_boost=2.5, partial_boost=1.5,
            )

        assert len(injected) == 1
        assert injected[0]["payload"]["source_id"] == "src-missing"
        # Floor score for exact: 0.15 * 2.5 = 0.375
        assert injected[0]["weighted_score"] == pytest.approx(0.15 * 2.5)
        assert injected[0]["field_scores"]["title_match"] == "exact"

    def test_partial_match_lower_floor_score(self):
        svc = _make_service()
        point = _make_point(source_id="src-2", field_value="policy")
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([point], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            injected = svc._fetch_field_match_docs(
                ["src-2"], {"src-2": "partial"}, "title",
                exact_boost=2.5, partial_boost=1.5,
            )

        assert injected[0]["weighted_score"] == pytest.approx(0.15 * 1.5)

    def test_skips_source_id_when_scroll_raises(self):
        svc = _make_service()
        mock_qdrant = Mock()
        mock_qdrant.scroll.side_effect = Exception("network error")

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            injected = svc._fetch_field_match_docs(
                ["src-err"], {"src-err": "exact"}, "title",
                exact_boost=2.5, partial_boost=1.5,
            )

        assert injected == []

    def test_skips_source_id_when_no_points_returned(self):
        svc = _make_service()
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([], None)  # no points

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            injected = svc._fetch_field_match_docs(
                ["src-empty"], {"src-empty": "exact"}, "title",
                exact_boost=2.5, partial_boost=1.5,
            )

        assert injected == []

    def test_floor_score_capped_at_one(self):
        """Even very high boost values must not push floor score above 1.0."""
        svc = _make_service()
        point = _make_point(source_id="src-1")
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([point], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            injected = svc._fetch_field_match_docs(
                ["src-1"], {"src-1": "exact"}, "title",
                exact_boost=100.0, partial_boost=1.5,
            )

        assert injected[0]["weighted_score"] <= 1.0


# ---------------------------------------------------------------------------
# _build_result_items
# ---------------------------------------------------------------------------

class TestBuildResultItems:
    def test_extracts_title_match_from_field_scores(self):
        svc = _make_service()
        top_results = [{
            "id": "pid-1",
            "payload": {
                "text": "some text",
                "title": "My Title",
                "summary": None,
                "tags": None,
                "metadata": {},
                "source_id": "src-1",
            },
            "weighted_score": 0.8,
            "field_scores": {"title": 0.8, "title_match": "exact"},
        }]
        items = svc._build_result_items(top_results)
        assert len(items) == 1
        assert items[0].title_match == "exact"
        # title_match must NOT remain in field_scores
        assert "title_match" not in items[0].field_scores

    def test_extracts_summary_match_from_field_scores(self):
        svc = _make_service()
        top_results = [{
            "id": "pid-1",
            "payload": {
                "text": "some text",
                "title": None,
                "summary": "My Summary",
                "tags": None,
                "metadata": {},
                "source_id": "src-1",
            },
            "weighted_score": 0.6,
            "field_scores": {"summary": 0.6, "summary_match": "partial"},
        }]
        items = svc._build_result_items(top_results)
        assert items[0].summary_match == "partial"
        assert "summary_match" not in items[0].field_scores

    def test_title_match_and_summary_match_none_when_absent(self):
        svc = _make_service()
        top_results = [{
            "id": "pid-1",
            "payload": {
                "text": "text",
                "title": None,
                "summary": None,
                "tags": None,
                "metadata": {},
                "source_id": "src-1",
            },
            "weighted_score": 0.5,
            "field_scores": {"text": 0.5},
        }]
        items = svc._build_result_items(top_results)
        assert items[0].title_match is None
        assert items[0].summary_match is None

    def test_field_scores_remains_pure_numeric_map(self):
        svc = _make_service()
        top_results = [{
            "id": "pid-1",
            "payload": {
                "text": "t",
                "title": "T",
                "summary": None,
                "tags": None,
                "metadata": {},
                "source_id": "src-1",
            },
            "weighted_score": 0.7,
            "field_scores": {
                "title": 0.7,
                "text": 0.5,
                "title_match": "partial",
                "summary_match": None,
            },
        }]
        items = svc._build_result_items(top_results)
        scores = items[0].field_scores
        assert "title_match" not in scores
        assert "summary_match" not in scores
        assert "title" in scores
        assert "text" in scores

    def test_skips_malformed_result_and_continues(self):
        """A result missing required payload keys should be skipped, not crash."""
        svc = _make_service()
        good = {
            "id": "pid-good",
            "payload": {
                "text": "good",
                "title": None,
                "summary": None,
                "tags": None,
                "metadata": {},
                "source_id": "src-good",
            },
            "weighted_score": 0.5,
            "field_scores": {},
        }
        # Bad result: missing 'payload' key entirely
        bad = {"id": "pid-bad", "weighted_score": 0.9}
        # _build_result_items should not crash on the bad item
        items = svc._build_result_items([bad, good])
        assert len(items) == 1
        assert items[0].id == "pid-good"


# ---------------------------------------------------------------------------
# _rank_results — RRF path (new in this PR)
# ---------------------------------------------------------------------------

class TestRankResultsRRFPath:
    def _make_qdrant_point(self, point_id, rrf_score, payload=None):
        p = Mock()
        p.id = point_id
        p.payload = payload or {"source_id": f"src-{point_id}", "text": "t",
                                "metadata": {}}
        p.score = rrf_score
        return p

    def test_rrf_score_used_directly(self):
        svc = _make_service()
        p1 = self._make_qdrant_point("p1", rrf_score=0.6)
        p2 = self._make_qdrant_point("p2", rrf_score=0.8)

        all_results = {"p1": p1, "p2": p2}
        field_scores = {"p1": {"rrf": 0.6}, "p2": {"rrf": 0.8}}

        ranked = svc._rank_results(
            all_results, field_scores,
            weights=svc.default_weights,
            search_fields=svc.priority_order,
        )

        assert ranked[0]["id"] == "p2"
        assert ranked[0]["weighted_score"] == pytest.approx(0.8)
        assert ranked[1]["id"] == "p1"
        assert ranked[1]["weighted_score"] == pytest.approx(0.6)

    def test_dense_weighted_sum_path(self):
        svc = _make_service()
        p1 = self._make_qdrant_point("p1", rrf_score=0.0)
        all_results = {"p1": p1}
        # Provide per-field scores (no "rrf" key → weighted sum path)
        field_scores = {"p1": {"title": 0.9, "text": 0.5}}
        weights = {"title": 0.36, "text": 0.27, "tags": 0.14,
                   "summary": 0.14, "metadata": 0.09}

        ranked = svc._rank_results(
            all_results, field_scores, weights=weights,
            search_fields=["title", "text"],
        )

        expected = 0.9 * 0.36 + 0.5 * 0.27
        assert ranked[0]["weighted_score"] == pytest.approx(expected)

    def test_score_not_capped_at_one_by_rank_results(self):
        """_rank_results must NOT cap at 1.0 — boost logic handles capping later."""
        svc = _make_service()
        p1 = self._make_qdrant_point("p1", 0.0)
        # Even though sum > 1.0, _rank_results should not cap it
        all_results = {"p1": p1}
        field_scores = {"p1": {"title": 1.0, "text": 1.0, "tags": 1.0}}
        weights = {"title": 0.36, "text": 0.27, "tags": 0.14}

        ranked = svc._rank_results(
            all_results, field_scores, weights=weights,
            search_fields=["title", "text", "tags"],
        )

        assert ranked[0]["weighted_score"] > 1.0  # NOT capped


# ---------------------------------------------------------------------------
# _get_field_match_sources — scroll-based
# ---------------------------------------------------------------------------

class TestGetFieldMatchSources:
    def test_returns_exact_match(self):
        svc = _make_service()
        point = _make_point(source_id="src-1", field_value="insurance",
                            field="title")
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([point], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            matches = svc._get_field_match_sources("insurance", None, "title")

        assert "src-1" in matches
        assert matches["src-1"] == "exact"

    def test_returns_partial_match(self):
        svc = _make_service()
        point = _make_point(source_id="src-2",
                            field_value="insurance policy document",
                            field="title")
        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([point], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            matches = svc._get_field_match_sources("insurance", None, "title")

        assert matches.get("src-2") == "partial"

    def test_empty_query_returns_empty_dict(self):
        svc = _make_service()
        mock_qdrant = Mock()

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            matches = svc._get_field_match_sources("   ", None, "title")

        mock_qdrant.scroll.assert_not_called()
        assert matches == {}

    def test_scroll_exception_is_non_fatal(self):
        svc = _make_service()
        mock_qdrant = Mock()
        mock_qdrant.scroll.side_effect = Exception("scroll error")

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            # Should not raise
            matches = svc._get_field_match_sources("test", None, "title")

        assert matches == {}

    def test_paginated_scroll_collects_all_pages(self):
        """If next_offset is non-None, scroll is called again for the next page."""
        svc = _make_service()
        p1 = _make_point(source_id="src-1", field_value="insurance", field="title",
                         point_id="pid-1")
        p2 = _make_point(source_id="src-2", field_value="insurance review",
                         field="title", point_id="pid-2")

        mock_qdrant = Mock()
        # First call returns p1 + a next_offset; second call returns p2 + None
        mock_qdrant.scroll.side_effect = [
            ([p1], "next-page-token"),
            ([p2], None),
        ]

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            matches = svc._get_field_match_sources("insurance", None, "title")

        assert "src-1" in matches
        assert "src-2" in matches
        assert mock_qdrant.scroll.call_count == 2

    def test_combined_filter_when_filter_conditions_has_must(self):
        """Existing filter conditions should be combined (not replaced) with field filter."""
        svc = _make_service()
        from qdrant_client import models as q_models

        existing_filter = q_models.Filter(
            must=[q_models.FieldCondition(
                key="metadata.company",
                match=q_models.MatchValue(value="Acme"),
            )]
        )

        mock_qdrant = Mock()
        mock_qdrant.scroll.return_value = ([], None)

        from app.services import prioritized_search_service as mod
        with patch.object(mod, "qdrant_client", mock_qdrant):
            svc._get_field_match_sources("insurance", existing_filter, "title")

        call_kwargs = mock_qdrant.scroll.call_args.kwargs
        combined = call_kwargs.get("scroll_filter")
        # Combined filter should have 2 must conditions (company + title)
        assert len(combined.must) == 2