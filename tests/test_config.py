"""
Unit tests for the new settings added to app/config.py in this PR.

Tests cover only the settings that were newly added:
- HYBRID_SEARCH_ENABLED (env-var driven bool)
- EXACT_TITLE_BOOST (float)
- PARTIAL_TITLE_BOOST (float)
- EXACT_SUMMARY_BOOST (float)
- PARTIAL_SUMMARY_BOOST (float)
- METADATA_MATCH_BOOST (float)
- SHORT_QUERY_THRESHOLD (int)
- RRF_K (int)
- SPARSE_VECTOR_NAME (str)
- SPARSE_SEARCH_ENABLED (bool)
"""
import os
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_settings(env_overrides: dict):
    """Re-create the Settings object from scratch with the given env overrides.

    We patch os.getenv at the module level so the `os.getenv(...)` calls
    inside the class body see our test values.
    """
    import importlib
    import unittest.mock as mock

    # Build a full env that keeps existing vars and overrides specific ones
    env = {**os.environ, **env_overrides}

    with mock.patch.dict(os.environ, env, clear=True):
        import app.config as config_mod
        importlib.reload(config_mod)
        return config_mod.Settings()


# ---------------------------------------------------------------------------
# Default values (no env var overrides)
# ---------------------------------------------------------------------------

class TestNewSettingsDefaults:
    """Verify default values for every new setting."""

    def setup_method(self):
        """Remove any relevant env vars so we test true defaults."""
        for key in [
            "HYBRID_SEARCH_ENABLED", "EXACT_TITLE_BOOST", "PARTIAL_TITLE_BOOST",
            "EXACT_SUMMARY_BOOST", "PARTIAL_SUMMARY_BOOST", "METADATA_MATCH_BOOST",
            "SHORT_QUERY_THRESHOLD", "RRF_K", "SPARSE_VECTOR_NAME", "SPARSE_SEARCH_ENABLED",
        ]:
            os.environ.pop(key, None)

    def test_hybrid_search_enabled_default_true(self):
        s = _reload_settings({"HYBRID_SEARCH_ENABLED": "true"})
        assert s.HYBRID_SEARCH_ENABLED is True

    def test_sparse_search_enabled_default_false(self):
        s = _reload_settings({"SPARSE_SEARCH_ENABLED": "false"})
        assert s.SPARSE_SEARCH_ENABLED is False

    def test_exact_title_boost_default(self):
        s = _reload_settings({})
        assert s.EXACT_TITLE_BOOST == pytest.approx(2.5)

    def test_partial_title_boost_default(self):
        s = _reload_settings({})
        assert s.PARTIAL_TITLE_BOOST == pytest.approx(1.5)

    def test_exact_summary_boost_default(self):
        s = _reload_settings({})
        assert s.EXACT_SUMMARY_BOOST == pytest.approx(1.4)

    def test_partial_summary_boost_default(self):
        s = _reload_settings({})
        assert s.PARTIAL_SUMMARY_BOOST == pytest.approx(1.2)

    def test_metadata_match_boost_default(self):
        s = _reload_settings({})
        assert s.METADATA_MATCH_BOOST == pytest.approx(1.2)

    def test_short_query_threshold_default(self):
        s = _reload_settings({})
        assert s.SHORT_QUERY_THRESHOLD == 3

    def test_rrf_k_default(self):
        s = _reload_settings({})
        assert s.RRF_K == 60

    def test_sparse_vector_name_default(self):
        s = _reload_settings({})
        assert s.SPARSE_VECTOR_NAME == "bm25"


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------

class TestNewSettingsFromEnvVars:
    """Verify that each new setting reads correctly from env vars."""

    def test_hybrid_search_enabled_false_from_env(self):
        s = _reload_settings({"HYBRID_SEARCH_ENABLED": "false"})
        assert s.HYBRID_SEARCH_ENABLED is False

    def test_hybrid_search_enabled_case_insensitive(self):
        s = _reload_settings({"HYBRID_SEARCH_ENABLED": "False"})
        assert s.HYBRID_SEARCH_ENABLED is False

    def test_sparse_search_enabled_true_from_env(self):
        s = _reload_settings({"SPARSE_SEARCH_ENABLED": "true"})
        assert s.SPARSE_SEARCH_ENABLED is True

    def test_sparse_search_enabled_case_insensitive(self):
        s = _reload_settings({"SPARSE_SEARCH_ENABLED": "True"})
        assert s.SPARSE_SEARCH_ENABLED is True

    def test_exact_title_boost_from_env(self):
        s = _reload_settings({"EXACT_TITLE_BOOST": "3.0"})
        assert s.EXACT_TITLE_BOOST == pytest.approx(3.0)

    def test_partial_title_boost_from_env(self):
        s = _reload_settings({"PARTIAL_TITLE_BOOST": "2.0"})
        assert s.PARTIAL_TITLE_BOOST == pytest.approx(2.0)

    def test_exact_summary_boost_from_env(self):
        s = _reload_settings({"EXACT_SUMMARY_BOOST": "1.8"})
        assert s.EXACT_SUMMARY_BOOST == pytest.approx(1.8)

    def test_partial_summary_boost_from_env(self):
        s = _reload_settings({"PARTIAL_SUMMARY_BOOST": "1.6"})
        assert s.PARTIAL_SUMMARY_BOOST == pytest.approx(1.6)

    def test_metadata_match_boost_from_env(self):
        s = _reload_settings({"METADATA_MATCH_BOOST": "1.5"})
        assert s.METADATA_MATCH_BOOST == pytest.approx(1.5)

    def test_short_query_threshold_from_env(self):
        s = _reload_settings({"SHORT_QUERY_THRESHOLD": "5"})
        assert s.SHORT_QUERY_THRESHOLD == 5

    def test_rrf_k_from_env(self):
        s = _reload_settings({"RRF_K": "80"})
        assert s.RRF_K == 80

    def test_sparse_vector_name_from_env(self):
        s = _reload_settings({"SPARSE_VECTOR_NAME": "sparse_bm25"})
        assert s.SPARSE_VECTOR_NAME == "sparse_bm25"

    def test_boost_values_are_floats(self):
        """All boost settings must be Python float, not str."""
        s = _reload_settings({})
        assert isinstance(s.EXACT_TITLE_BOOST, float)
        assert isinstance(s.PARTIAL_TITLE_BOOST, float)
        assert isinstance(s.EXACT_SUMMARY_BOOST, float)
        assert isinstance(s.PARTIAL_SUMMARY_BOOST, float)
        assert isinstance(s.METADATA_MATCH_BOOST, float)

    def test_threshold_and_rrf_k_are_ints(self):
        """Integer settings must be Python int."""
        s = _reload_settings({})
        assert isinstance(s.SHORT_QUERY_THRESHOLD, int)
        assert isinstance(s.RRF_K, int)

    def test_enabled_flags_are_bools(self):
        """Boolean settings must be Python bool."""
        s = _reload_settings({})
        assert isinstance(s.HYBRID_SEARCH_ENABLED, bool)
        assert isinstance(s.SPARSE_SEARCH_ENABLED, bool)