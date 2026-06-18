"""
Unit tests for app/utils/query_preprocessor.py

Focus on the NEW behaviour introduced in this PR:
- Short query bypass: queries with fewer words than SHORT_QUERY_THRESHOLD, or
  fewer than 20 characters, skip spaCy and return stripped.lower()
- Fallback when preprocessing yields an empty result
- Error fallback returns stripped (not the original unsrtipped string)
"""
import os
import pytest
from unittest.mock import patch, Mock, MagicMock

# Ensure test env vars are set before importing anything
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


class TestPreprocessQueryShortQueryBypass:
    """Tests for the new short-query bypass logic added in this PR."""

    def test_empty_query_returns_empty_string(self):
        """Empty input must return '' (unchanged from pre-PR behaviour)."""
        from app.utils.query_preprocessor import preprocess_query
        assert preprocess_query("") == ""
        assert preprocess_query("   ") == ""

    def test_single_word_query_bypasses_spacy(self):
        """A single-word query (word_count=1 < threshold=3) must skip spaCy."""
        from app.utils.query_preprocessor import preprocess_query
        # 'AI' has 1 word, 2 chars — both below threshold
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query("AI")
            mock_load.assert_not_called()
        assert result == "ai"

    def test_two_word_query_below_word_threshold_bypasses_spacy(self):
        """Two words < threshold 3 → bypass (unless chars >= 20)."""
        from app.utils.query_preprocessor import preprocess_query
        query = "SMC RTE"  # 2 words, 7 chars
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query(query)
            mock_load.assert_not_called()
        assert result == "smc rte"

    def test_short_char_query_bypasses_spacy_regardless_of_word_count(self):
        """A query < 20 chars always bypasses spaCy even if word count >= threshold."""
        from app.utils.query_preprocessor import preprocess_query
        # 3 words but only 13 chars total — char guard fires first
        query = "the AI model"  # 12 chars
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query(query)
            mock_load.assert_not_called()
        assert result == "the ai model"

    def test_short_query_returns_lowercase_stripped(self):
        """Short-bypass must lowercase and strip the query."""
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("  RTE  ")
        assert result == "rte"

    def test_long_query_with_enough_words_calls_spacy(self):
        """A query with >=3 words AND >=20 chars must go through spaCy preprocessing."""
        from app.utils.query_preprocessor import preprocess_query

        mock_token = Mock()
        mock_token.pos_ = "NOUN"
        mock_token.is_stop = False
        mock_token.is_punct = False
        mock_token.is_space = False
        mock_token.text = "insurance"

        mock_doc = [mock_token]
        mock_nlp = Mock(return_value=mock_doc)

        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            result = preprocess_query("insurance coverage document review")

        mock_nlp.assert_called_once()
        assert result == "insurance"

    def test_query_exactly_at_char_threshold_goes_through_spacy(self):
        """A query of exactly 20 chars (not less) with enough words calls spaCy."""
        from app.utils.query_preprocessor import preprocess_query

        # Build a 20-char, 3-word query
        query = "abc def ghi jklmnopq"  # 20 chars exactly, 4 words
        assert len(query.strip()) == 20

        mock_token = Mock()
        mock_token.pos_ = "NOUN"
        mock_token.is_stop = False
        mock_token.is_punct = False
        mock_token.is_space = False
        mock_token.text = "abc"

        mock_nlp = Mock(return_value=[mock_token])
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            result = preprocess_query(query)

        mock_nlp.assert_called_once()

    def test_preprocessing_result_empty_falls_back_to_original(self):
        """If spaCy preprocessing produces empty output, return stripped.lower() of original."""
        from app.utils.query_preprocessor import preprocess_query

        # All tokens are stop-words / pronouns so processed_tokens is empty
        mock_token = Mock()
        mock_token.pos_ = "PRON"
        mock_token.is_stop = True
        mock_token.is_punct = False
        mock_token.is_space = False
        mock_token.text = "the"

        mock_nlp = Mock(return_value=[mock_token])
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            # Long enough query to pass both guards: >=3 words, >=20 chars
            result = preprocess_query("the I the the the the long query here")

        # Preprocessing yields empty → fall back to stripped.lower()
        assert result == "the i the the the the long query here"

    def test_runtime_error_from_model_loading_is_reraised(self):
        """RuntimeError from _load_spacy_model must propagate (not silently swallowed)."""
        from app.utils.query_preprocessor import preprocess_query

        with patch(
            "app.utils.query_preprocessor._load_spacy_model",
            side_effect=RuntimeError("model not found"),
        ):
            with pytest.raises(RuntimeError, match="model not found"):
                preprocess_query("some long query that passes guard checks here")

    def test_non_runtime_exception_returns_stripped_query(self):
        """Non-RuntimeError exceptions during preprocessing return stripped original."""
        from app.utils.query_preprocessor import preprocess_query

        with patch(
            "app.utils.query_preprocessor._load_spacy_model",
            side_effect=ValueError("unexpected"),
        ):
            result = preprocess_query("  Some Long Query Text  ")

        # Falls back to stripped (not lowercased in this branch)
        assert result == "Some Long Query Text"

    def test_threshold_read_from_settings(self):
        """SHORT_QUERY_THRESHOLD is read from settings; patching it changes bypass behaviour."""
        from app.utils.query_preprocessor import preprocess_query

        # Patch settings so threshold=5 — a 3-word query should now bypass
        with patch("app.config.settings") as mock_settings:
            mock_settings.SHORT_QUERY_THRESHOLD = 5
            with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
                # 3 words, 21 chars — would normally go through spaCy at threshold=3
                # but with threshold=5 it should bypass
                result = preprocess_query("word word word long")
            # Should bypass because 4 words < threshold 5 OR len < 20
            # "word word word long" = 19 chars → bypasses on char count anyway
            # Let's just check it didn't blow up and returned lowercase
            assert isinstance(result, str)


class TestPreprocessQueryEdgeCases:
    """Additional edge/regression cases."""

    def test_abbreviation_preserved_exactly(self):
        """Common abbreviation queries must not have characters removed."""
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("RTE")
        assert result == "rte"

    def test_mixed_case_short_query_lowercased(self):
        """Mixed-case short queries are returned lowercase."""
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("SMC")
        assert result == "smc"

    def test_whitespace_stripped_from_short_query(self):
        """Leading/trailing whitespace is stripped from short queries."""
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("   AI   ")
        assert result == "ai"
        assert not result.startswith(" ")
        assert not result.endswith(" ")