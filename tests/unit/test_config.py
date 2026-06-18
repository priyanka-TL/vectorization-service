"""
Unit tests for app/config.py

Focus on the new Settings fields added in this PR:
- HYBRID_SEARCH_ENABLED (bool, from env, default True)
- EXACT_TITLE_BOOST (float, default 2.5)
- PARTIAL_TITLE_BOOST (float, default 1.5)
- EXACT_SUMMARY_BOOST (float, default 1.4)
- PARTIAL_SUMMARY_BOOST (float, default 1.2)
- METADATA_MATCH_BOOST (float, default 1.2)
- SHORT_QUERY_THRESHOLD (int, default 3)
- RRF_K (int, default 60)
- SPARSE_VECTOR_NAME (str, default "bm25")
- SPARSE_SEARCH_ENABLED (bool, from env, default False)
"""
import os
import pytest


class TestSettingsDefaults:
    """Verify default values for new settings fields without env overrides."""

    def _make_settings(self, env_overrides=None):
        """Instantiate a fresh Settings object with optional env variable overrides."""
        env_overrides = env_overrides or {}
        original = {}

        # Store originals, then set overrides
        for key, val in env_overrides.items():
            original[key] = os.environ.get(key)
            os.environ[key] = val

        try:
            # Remove old singleton before creating a fresh one
            import importlib
            import app.config as config_mod
            # Force a fresh Settings instance
            from app.config import Settings
            s = Settings()
        finally:
            # Restore originals
            for key, old_val in original.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val

        return s

    def test_hybrid_search_enabled_default_true(self):
        env = {k: v for k, v in os.environ.items()}
        env.pop("HYBRID_SEARCH_ENABLED", None)
        s = self._make_settings()
        # Default from os.getenv("HYBRID_SEARCH_ENABLED", "true") → True
        assert s.HYBRID_SEARCH_ENABLED is True

    def test_sparse_search_enabled_default_false(self):
        env = {k: v for k, v in os.environ.items()}
        env.pop("SPARSE_SEARCH_ENABLED", None)
        s = self._make_settings()
        assert s.SPARSE_SEARCH_ENABLED is False

    def test_exact_title_boost_default(self):
        s = self._make_settings()
        assert s.EXACT_TITLE_BOOST == pytest.approx(2.5)

    def test_partial_title_boost_default(self):
        s = self._make_settings()
        assert s.PARTIAL_TITLE_BOOST == pytest.approx(1.5)

    def test_exact_summary_boost_default(self):
        s = self._make_settings()
        assert s.EXACT_SUMMARY_BOOST == pytest.approx(1.4)

    def test_partial_summary_boost_default(self):
        s = self._make_settings()
        assert s.PARTIAL_SUMMARY_BOOST == pytest.approx(1.2)

    def test_metadata_match_boost_default(self):
        s = self._make_settings()
        assert s.METADATA_MATCH_BOOST == pytest.approx(1.2)

    def test_short_query_threshold_default(self):
        s = self._make_settings()
        assert s.SHORT_QUERY_THRESHOLD == 3

    def test_rrf_k_default(self):
        s = self._make_settings()
        assert s.RRF_K == 60

    def test_sparse_vector_name_default(self):
        s = self._make_settings()
        assert s.SPARSE_VECTOR_NAME == "bm25"


class TestSettingsEnvOverrides:
    """Verify that environment variables correctly override new settings fields."""

    def _make_settings_with_env(self, overrides: dict):
        """Temporarily set env vars and create a fresh Settings instance."""
        original = {}
        for key, val in overrides.items():
            original[key] = os.environ.get(key)
            os.environ[key] = val
        try:
            from app.config import Settings
            return Settings()
        finally:
            for key, old_val in original.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val

    def test_hybrid_search_enabled_overridden_to_false(self):
        s = self._make_settings_with_env({"HYBRID_SEARCH_ENABLED": "false"})
        assert s.HYBRID_SEARCH_ENABLED is False

    def test_hybrid_search_enabled_overridden_to_true(self):
        s = self._make_settings_with_env({"HYBRID_SEARCH_ENABLED": "true"})
        assert s.HYBRID_SEARCH_ENABLED is True

    def test_sparse_search_enabled_overridden_to_true(self):
        s = self._make_settings_with_env({"SPARSE_SEARCH_ENABLED": "true"})
        assert s.SPARSE_SEARCH_ENABLED is True

    def test_sparse_search_enabled_case_insensitive(self):
        s = self._make_settings_with_env({"SPARSE_SEARCH_ENABLED": "TRUE"})
        assert s.SPARSE_SEARCH_ENABLED is True

    def test_hybrid_search_enabled_case_insensitive(self):
        s = self._make_settings_with_env({"HYBRID_SEARCH_ENABLED": "False"})
        assert s.HYBRID_SEARCH_ENABLED is False

    def test_exact_title_boost_overridden(self):
        s = self._make_settings_with_env({"EXACT_TITLE_BOOST": "3.0"})
        assert s.EXACT_TITLE_BOOST == pytest.approx(3.0)

    def test_partial_title_boost_overridden(self):
        s = self._make_settings_with_env({"PARTIAL_TITLE_BOOST": "2.0"})
        assert s.PARTIAL_TITLE_BOOST == pytest.approx(2.0)

    def test_exact_summary_boost_overridden(self):
        s = self._make_settings_with_env({"EXACT_SUMMARY_BOOST": "1.8"})
        assert s.EXACT_SUMMARY_BOOST == pytest.approx(1.8)

    def test_partial_summary_boost_overridden(self):
        s = self._make_settings_with_env({"PARTIAL_SUMMARY_BOOST": "1.5"})
        assert s.PARTIAL_SUMMARY_BOOST == pytest.approx(1.5)

    def test_short_query_threshold_overridden(self):
        s = self._make_settings_with_env({"SHORT_QUERY_THRESHOLD": "5"})
        assert s.SHORT_QUERY_THRESHOLD == 5

    def test_rrf_k_overridden(self):
        s = self._make_settings_with_env({"RRF_K": "80"})
        assert s.RRF_K == 80

    def test_sparse_vector_name_overridden(self):
        s = self._make_settings_with_env({"SPARSE_VECTOR_NAME": "custom_sparse"})
        assert s.SPARSE_VECTOR_NAME == "custom_sparse"


class TestSettingsTypeCorrectness:
    """Verify that new settings have the correct Python types."""

    def _settings(self):
        from app.config import Settings
        return Settings()

    def test_hybrid_search_enabled_is_bool(self):
        assert isinstance(self._settings().HYBRID_SEARCH_ENABLED, bool)

    def test_sparse_search_enabled_is_bool(self):
        assert isinstance(self._settings().SPARSE_SEARCH_ENABLED, bool)

    def test_exact_title_boost_is_float(self):
        assert isinstance(self._settings().EXACT_TITLE_BOOST, float)

    def test_partial_title_boost_is_float(self):
        assert isinstance(self._settings().PARTIAL_TITLE_BOOST, float)

    def test_exact_summary_boost_is_float(self):
        assert isinstance(self._settings().EXACT_SUMMARY_BOOST, float)

    def test_partial_summary_boost_is_float(self):
        assert isinstance(self._settings().PARTIAL_SUMMARY_BOOST, float)

    def test_short_query_threshold_is_int(self):
        assert isinstance(self._settings().SHORT_QUERY_THRESHOLD, int)

    def test_rrf_k_is_int(self):
        assert isinstance(self._settings().RRF_K, int)

    def test_sparse_vector_name_is_str(self):
        assert isinstance(self._settings().SPARSE_VECTOR_NAME, str)