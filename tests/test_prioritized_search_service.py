"""
Unit tests for app/services/prioritized_search_service.py

Tests focus on new methods/changes introduced in this PR:
- _classify_text_match (new static method)
- _build_result_items — extracts title_match / summary_match from field_scores
- _apply_field_boost — multiplier logic, cap at 1.0, re-sort, records match key
- _supplement_matches_from_results — in-memory pass, no duplicate adds
- _fetch_field_match_docs — floor score, scroll-based, skip on empty
- _rank_results — RRF path vs weighted-sum path (score not capped at 1.0)
- Title wrappers (_get_title_match_sources, _apply_title_boost, _fetch_title_match_docs)
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service():
    """Instantiate PrioritizedSearchService with mocked Qdrant + embedding."""
    with patch("app.services.prioritized_search_service.qdrant_client"):
        from app.services.prioritized_search_service import PrioritizedSearchService
        return PrioritizedSearchService()


def _make_result(source_id, weighted_score, field_scores=None, payload_extra=None):
    payload = {"source_id": source_id, "text": "txt", "metadata": {}}
    if payload_extra:
        payload.update(payload_extra)
    return {
        "id": source_id + "-chunk",
        "payload": payload,
        "weighted_score": weighted_score,
        "field_scores": field_scores or {},
    }


# ---------------------------------------------------------------------------
# _classify_text_match
# ---------------------------------------------------------------------------

class TestClassifyTextMatch:
    """Tests for the static _classify_text_match method."""

    def setup_method(self):
        self.svc = _make_service()

    def test_exact_match_returns_exact(self):
        assert self.svc._classify_text_match("insurance", "insurance") == "exact"

    def test_partial_prefix_match_returns_partial(self):
        assert self.svc._classify_text_match("insur", "insurance policy") == "partial"

    def test_partial_infix_match_returns_partial(self):
        """Mid/infix matches (query inside field, not at start) → 'partial'."""
        assert self.svc._classify_text_match("sur", "insurance") == "partial"

    def test_no_match_returns_none(self):
        assert self.svc._classify_text_match("xyz", "insurance policy") is None

    def test_empty_field_returns_none(self):
        assert self.svc._classify_text_match("query", "") is None

    def test_query_longer_than_field_no_match(self):
        assert self.svc._classify_text_match("insurance policy docs", "insurance") is None

    def test_case_sensitive_by_default(self):
        """Method works on pre-lowercased strings; UPPER should not match lower."""
        assert self.svc._classify_text_match("INSURANCE", "insurance") is None

    def test_both_same_after_lower(self):
        """Callers should pre-lower; confirm exact when identical."""
        q = "risk management"
        assert self.svc._classify_text_match(q, q) == "exact"

    def test_whitespace_padding_counts_as_partial(self):
        """Field with surrounding words → partial, not exact."""
        assert self.svc._classify_text_match("risk", "risk management") == "partial"


# ---------------------------------------------------------------------------
# _build_result_items
# ---------------------------------------------------------------------------

class TestBuildResultItems:
    """Tests for _build_result_items — extraction of match types."""

    def setup_method(self):
        self.svc = _make_service()

    def _raw(self, source_id="s1", score=0.8, field_scores=None):
        return {
            "id": "chunk-1",
            "payload": {
                "source_id": source_id,
                "text": "text",
                "title": "Title",
                "summary": "Summary",
                "tags": [],
                "metadata": {},
            },
            "weighted_score": score,
            "field_scores": field_scores or {},
        }

    def test_title_match_extracted_from_field_scores(self):
        raw = self._raw(field_scores={"title": 0.9, "title_match": "exact"})
        items = self.svc._build_result_items([raw])
        assert items[0].title_match == "exact"
        assert "title_match" not in items[0].field_scores

    def test_summary_match_extracted_from_field_scores(self):
        raw = self._raw(field_scores={"summary": 0.7, "summary_match": "partial"})
        items = self.svc._build_result_items([raw])
        assert items[0].summary_match == "partial"
        assert "summary_match" not in items[0].field_scores

    def test_no_match_types_when_absent(self):
        raw = self._raw(field_scores={"title": 0.5})
        items = self.svc._build_result_items([raw])
        assert items[0].title_match is None
        assert items[0].summary_match is None

    def test_field_scores_without_match_keys(self):
        """Pure numeric field scores should remain in field_scores."""
        raw = self._raw(field_scores={"title": 0.9, "text": 0.6})
        items = self.svc._build_result_items([raw])
        assert "title" in items[0].field_scores
        assert "text" in items[0].field_scores

    def test_score_set_from_weighted_score(self):
        raw = self._raw(score=0.77)
        items = self.svc._build_result_items([raw])
        assert items[0].score == pytest.approx(0.77)

    def test_bad_result_data_skipped(self):
        """Malformed result dicts should be silently skipped."""
        bad = {"id": "bad", "payload": None, "weighted_score": 0.5, "field_scores": {}}
        items = self.svc._build_result_items([bad])
        assert items == []


# ---------------------------------------------------------------------------
# _apply_field_boost
# ---------------------------------------------------------------------------

class TestApplyFieldBoost:
    """Tests for _apply_field_boost."""

    def setup_method(self):
        self.svc = _make_service()

    def test_exact_match_applies_exact_boost(self):
        results = [_make_result("s1", 0.4, {})]
        matches = {"s1": "exact"}
        boosted = self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(min(0.4 * 2.5, 1.0))

    def test_partial_match_applies_partial_boost(self):
        results = [_make_result("s1", 0.4, {})]
        matches = {"s1": "partial"}
        boosted = self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(min(0.4 * 1.5, 1.0))

    def test_score_capped_at_1_0(self):
        results = [_make_result("s1", 0.9, {})]
        matches = {"s1": "exact"}
        boosted = self.svc._apply_field_boost(results, matches, "title", 3.0, 2.0)
        assert boosted[0]["weighted_score"] == pytest.approx(1.0)

    def test_no_match_score_unchanged(self):
        results = [_make_result("s1", 0.5, {})]
        matches = {}  # no match for s1
        boosted = self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        assert boosted[0]["weighted_score"] == pytest.approx(0.5)

    def test_match_type_recorded_in_field_scores(self):
        results = [_make_result("s1", 0.4, {})]
        matches = {"s1": "partial"}
        self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        assert results[0]["field_scores"]["title_match"] == "partial"

    def test_no_match_sets_match_key_to_none(self):
        results = [_make_result("s1", 0.4, {})]
        self.svc._apply_field_boost(results, {}, "title", 2.5, 1.5)
        assert results[0]["field_scores"].get("title_match") is None

    def test_results_sorted_descending_after_boost(self):
        results = [
            _make_result("s1", 0.3, {}),
            _make_result("s2", 0.5, {}),
        ]
        matches = {"s1": "exact"}  # s1 boosted to 0.3*2.5=0.75 > s2=0.5
        boosted = self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        assert boosted[0]["payload"]["source_id"] == "s1"

    def test_summary_field_uses_summary_match_key(self):
        results = [_make_result("s1", 0.4, {})]
        matches = {"s1": "exact"}
        self.svc._apply_field_boost(results, matches, "summary", 1.4, 1.2)
        assert "summary_match" in results[0]["field_scores"]

    def test_multiple_results_only_matched_boosted(self):
        results = [
            _make_result("s1", 0.4, {}),
            _make_result("s2", 0.4, {}),
        ]
        matches = {"s1": "partial"}
        boosted = self.svc._apply_field_boost(results, matches, "title", 2.5, 1.5)
        s1 = next(r for r in boosted if r["payload"]["source_id"] == "s1")
        s2 = next(r for r in boosted if r["payload"]["source_id"] == "s2")
        assert s1["weighted_score"] > s2["weighted_score"]


# ---------------------------------------------------------------------------
# _supplement_matches_from_results
# ---------------------------------------------------------------------------

class TestSupplementMatchesFromResults:
    """Tests for _supplement_matches_from_results."""

    def setup_method(self):
        self.svc = _make_service()

    def _result_with_title(self, source_id, title):
        return {
            "payload": {"source_id": source_id, "title": title},
            "weighted_score": 0.5,
            "field_scores": {},
        }

    def test_infix_match_added_to_matches(self):
        matches = {}
        results = [self._result_with_title("s1", "Insurance and Risk Management")]
        self.svc._supplement_matches_from_results("sur", results, "title", matches)
        assert "s1" in matches
        assert matches["s1"] == "partial"

    def test_exact_match_added(self):
        matches = {}
        results = [self._result_with_title("s1", "risk management")]
        self.svc._supplement_matches_from_results("risk management", results, "title", matches)
        assert matches["s1"] == "exact"

    def test_already_matched_source_not_overwritten(self):
        matches = {"s1": "exact"}
        results = [self._result_with_title("s1", "risk management document here")]
        self.svc._supplement_matches_from_results("risk management", results, "title", matches)
        assert matches["s1"] == "exact"  # not downgraded to partial

    def test_no_match_leaves_matches_unchanged(self):
        matches = {}
        results = [self._result_with_title("s1", "something unrelated")]
        self.svc._supplement_matches_from_results("insurance", results, "title", matches)
        assert matches == {}

    def test_empty_query_returns_without_changes(self):
        matches = {}
        results = [self._result_with_title("s1", "some title")]
        self.svc._supplement_matches_from_results("", results, "title", matches)
        assert matches == {}

    def test_missing_source_id_skipped(self):
        matches = {}
        results = [{"payload": {"title": "insurance title"}, "weighted_score": 0.5, "field_scores": {}}]
        self.svc._supplement_matches_from_results("insurance", results, "title", matches)
        assert matches == {}


# ---------------------------------------------------------------------------
# _fetch_field_match_docs
# ---------------------------------------------------------------------------

class TestFetchFieldMatchDocs:
    """Tests for _fetch_field_match_docs."""

    def setup_method(self):
        self.svc = _make_service()

    def _make_point(self, source_id="s1"):
        p = Mock()
        p.id = source_id + "-chunk"
        p.payload = {"source_id": source_id, "text": "doc text", "metadata": {}}
        return p

    def test_returns_injected_doc_with_floor_score(self):
        point = self._make_point("s1")
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.return_value = ([point], None)
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "exact"}, "title", 2.5, 1.5
            )

        assert len(injected) == 1
        # FLOOR_SCORE = 0.15; exact → 0.15 * 2.5 = 0.375
        assert injected[0]["weighted_score"] == pytest.approx(0.15 * 2.5)

    def test_partial_match_uses_partial_boost(self):
        point = self._make_point("s1")
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.return_value = ([point], None)
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "partial"}, "title", 2.5, 1.5
            )

        assert injected[0]["weighted_score"] == pytest.approx(0.15 * 1.5)

    def test_score_capped_at_1_0(self):
        point = self._make_point("s1")
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.return_value = ([point], None)
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "exact"}, "title", 100.0, 50.0  # very large boosts
            )

        assert injected[0]["weighted_score"] == pytest.approx(1.0)

    def test_empty_scroll_skips_source(self):
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.return_value = ([], None)
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "exact"}, "title", 2.5, 1.5
            )

        assert injected == []

    def test_scroll_exception_skips_source(self):
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.side_effect = RuntimeError("scroll error")
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "exact"}, "title", 2.5, 1.5
            )

        assert injected == []

    def test_match_key_in_field_scores(self):
        point = self._make_point("s1")
        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.return_value = ([point], None)
            injected = self.svc._fetch_field_match_docs(
                ["s1"], {"s1": "partial"}, "title", 2.5, 1.5
            )

        assert injected[0]["field_scores"]["title_match"] == "partial"

    def test_multiple_source_ids_processed(self):
        def scroll_side_effect(**kwargs):
            filt = kwargs.get("scroll_filter")
            source_id = filt.must[0].match.value
            p = self._make_point(source_id)
            return ([p], None)

        with patch("app.services.prioritized_search_service.qdrant_client") as mock_q:
            mock_q.scroll.side_effect = scroll_side_effect
            injected = self.svc._fetch_field_match_docs(
                ["s1", "s2"],
                {"s1": "exact", "s2": "partial"},
                "title", 2.5, 1.5
            )

        assert len(injected) == 2


# ---------------------------------------------------------------------------
# _rank_results — RRF path vs weighted-sum path
# ---------------------------------------------------------------------------

class TestRankResults:
    """Tests for _rank_results — PR changes: RRF path + no premature cap."""

    def setup_method(self):
        self.svc = _make_service()

    def _make_qdrant_point(self, pid, payload):
        p = Mock()
        p.id = pid
        p.payload = payload
        return p

    def test_rrf_path_uses_rrf_score_directly(self):
        """When field_scores contains 'rrf', weighted_score == rrf score."""
        point = self._make_qdrant_point("p1", {"source_id": "s1"})
        all_results = {"p1": point}
        field_scores = {"p1": {"rrf": 0.654}}
        ranked = self.svc._rank_results(all_results, field_scores, {}, [])
        assert ranked[0]["weighted_score"] == pytest.approx(0.654)

    def test_dense_path_uses_weighted_sum(self):
        point = self._make_qdrant_point("p1", {"source_id": "s1"})
        all_results = {"p1": point}
        field_scores = {"p1": {"title": 0.8, "text": 0.6}}
        weights = {"title": 0.4, "text": 0.3}
        search_fields = ["title", "text"]
        ranked = self.svc._rank_results(all_results, field_scores, weights, search_fields)
        expected = 0.8 * 0.4 + 0.6 * 0.3
        assert ranked[0]["weighted_score"] == pytest.approx(expected)

    def test_dense_score_not_capped_at_1_0(self):
        """Dense path must NOT cap at 1.0 (boost may push it above)."""
        point = self._make_qdrant_point("p1", {"source_id": "s1"})
        all_results = {"p1": point}
        field_scores = {"p1": {"title": 1.0}}
        weights = {"title": 1.5}  # weight > 1.0 on purpose
        ranked = self.svc._rank_results(all_results, field_scores, weights, ["title"])
        assert ranked[0]["weighted_score"] == pytest.approx(1.5)

    def test_results_sorted_descending(self):
        p1 = self._make_qdrant_point("p1", {"source_id": "s1"})
        p2 = self._make_qdrant_point("p2", {"source_id": "s2"})
        all_results = {"p1": p1, "p2": p2}
        field_scores = {
            "p1": {"title": 0.5},
            "p2": {"title": 0.9},
        }
        weights = {"title": 1.0}
        ranked = self.svc._rank_results(all_results, field_scores, weights, ["title"])
        assert ranked[0]["id"] == "p2"
        assert ranked[1]["id"] == "p1"

    def test_missing_field_in_scores_contributes_zero(self):
        """Fields absent from field_score_dict add 0 to weighted_score."""
        point = self._make_qdrant_point("p1", {"source_id": "s1"})
        all_results = {"p1": point}
        field_scores = {"p1": {"title": 0.8}}
        weights = {"title": 0.4, "text": 0.3}  # 'text' not in field_scores
        ranked = self.svc._rank_results(all_results, field_scores, weights, ["title", "text"])
        assert ranked[0]["weighted_score"] == pytest.approx(0.8 * 0.4)


# ---------------------------------------------------------------------------
# Title wrappers (backward-compat)
# ---------------------------------------------------------------------------

class TestTitleWrappers:
    """Tests confirming wrapper methods delegate to generic field methods."""

    def setup_method(self):
        self.svc = _make_service()

    def test_apply_title_boost_delegates_to_apply_field_boost(self):
        results = [_make_result("s1", 0.4, {})]
        matches = {"s1": "exact"}
        with patch.object(self.svc, "_apply_field_boost", wraps=self.svc._apply_field_boost) as m:
            self.svc._apply_title_boost(results, matches)
        m.assert_called_once()
        call_args = m.call_args
        assert call_args[0][2] == "title"  # field argument

    def test_get_title_match_sources_delegates_to_get_field_match_sources(self):
        with patch.object(self.svc, "_get_field_match_sources", return_value={}) as m:
            self.svc._get_title_match_sources("query", None)
        m.assert_called_once_with("query", None, "title")

    def test_fetch_title_match_docs_delegates_to_fetch_field_match_docs(self):
        with patch.object(self.svc, "_fetch_field_match_docs", return_value=[]) as m:
            self.svc._fetch_title_match_docs(["s1"], {"s1": "exact"})
        m.assert_called_once()
        call_args = m.call_args[0]
        assert call_args[2] == "title"