"""
Unit tests for app/utils/query_preprocessor.py

Tests cover:
- Empty / whitespace-only input → empty string
- Short queries (word count < threshold) → lowercased original (bypass spaCy)
- Short queries by character count (< 20 chars) → lowercased original
- Normal queries go through spaCy pipeline
- Preprocessing produces empty result → falls back to stripped.lower()
- RuntimeError from spaCy model loading is re-raised
- Other exceptions from preprocessing → returns stripped original
- is_spacy_model_available helper
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import os

# Ensure test env vars are set before any app import
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_CACHE_ENABLED", "False")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_nlp_singleton():
    """Reset the module-level spaCy singleton before every test."""
    import app.utils.query_preprocessor as mod
    original = mod._nlp_model
    mod._nlp_model = None
    yield
    mod._nlp_model = original


def _make_spacy_doc(tokens):
    """Build a lightweight mock spaCy doc from a list of token specs.

    Each spec is a dict: {text, pos_, is_stop, is_punct, is_space}.
    """
    mock_tokens = []
    for spec in tokens:
        t = Mock()
        t.text = spec["text"]
        t.pos_ = spec.get("pos_", "NOUN")
        t.is_stop = spec.get("is_stop", False)
        t.is_punct = spec.get("is_punct", False)
        t.is_space = spec.get("is_space", False)
        mock_tokens.append(t)

    doc = MagicMock()
    doc.__iter__ = Mock(return_value=iter(mock_tokens))
    return doc


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string_returns_empty(self):
        from app.utils.query_preprocessor import preprocess_query
        assert preprocess_query("") == ""

    def test_whitespace_only_returns_empty(self):
        from app.utils.query_preprocessor import preprocess_query
        assert preprocess_query("   ") == ""

    def test_newline_only_returns_empty(self):
        from app.utils.query_preprocessor import preprocess_query
        assert preprocess_query("\n\t") == ""


# ---------------------------------------------------------------------------
# Short query bypass (word count < threshold OR char count < 20)
# ---------------------------------------------------------------------------

class TestShortQueryBypass:
    """Queries below SHORT_QUERY_THRESHOLD words (default 3) or shorter than
    20 chars bypass spaCy and are returned as lowercased original text."""

    def test_single_word_query_bypasses_spacy(self):
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query("SMC")
        mock_load.assert_not_called()
        assert result == "smc"

    def test_two_word_short_query_bypasses_spacy(self):
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query("AI policy")
        mock_load.assert_not_called()
        assert result == "ai policy"

    def test_short_char_query_bypasses_spacy(self):
        """Even 3 words but fewer than 20 chars → bypass."""
        from app.utils.query_preprocessor import preprocess_query
        # "a b c" = 3 words (>= threshold), but only 5 chars (<20) → bypass
        with patch("app.utils.query_preprocessor._load_spacy_model") as mock_load:
            result = preprocess_query("a b c")
        mock_load.assert_not_called()
        assert result == "a b c"

    def test_short_query_preserves_mixed_case_as_lower(self):
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("RTE Act")
        assert result == "rte act"

    def test_short_query_strips_leading_trailing_whitespace(self):
        from app.utils.query_preprocessor import preprocess_query
        with patch("app.utils.query_preprocessor._load_spacy_model"):
            result = preprocess_query("  SMC  ")
        assert result == "smc"

    def test_exactly_threshold_words_with_enough_chars_goes_through_spacy(self):
        """Exactly threshold words AND >= 20 chars should pass through spaCy."""
        # Default threshold = 3; use a 3-word query with 20+ chars
        long_query = "information security policy"  # 3 words, 28 chars
        assert len(long_query) >= 20

        mock_doc = _make_spacy_doc([
            {"text": "information"},
            {"text": "security"},
            {"text": "policy"},
        ])
        mock_nlp = Mock(return_value=mock_doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            result = preprocess_query(long_query)
        assert result == "information security policy"


# ---------------------------------------------------------------------------
# Normal (long) query processing
# ---------------------------------------------------------------------------

class TestNormalQueryProcessing:
    """Queries above the threshold are processed by spaCy."""

    def test_stop_words_removed(self):
        """Tokens flagged as stop words should be excluded."""
        doc = _make_spacy_doc([
            {"text": "the", "is_stop": True},
            {"text": "document", "is_stop": False},
            {"text": "is", "is_stop": True},
            {"text": "important", "is_stop": False},
        ])
        mock_nlp = Mock(return_value=doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            # Use a long enough query to bypass short-query logic
            result = preprocess_query("the document is important indeed here")
        assert "the" not in result
        assert "is" not in result
        assert "document" in result
        assert "important" in result

    def test_pronouns_removed(self):
        """Tokens with POS == PRON should be excluded."""
        doc = _make_spacy_doc([
            {"text": "I", "pos_": "PRON"},
            {"text": "need", "pos_": "VERB"},
            {"text": "information", "pos_": "NOUN"},
            {"text": "about", "pos_": "ADP"},
            {"text": "security", "pos_": "NOUN"},
            {"text": "policies", "pos_": "NOUN"},
        ])
        mock_nlp = Mock(return_value=doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            result = preprocess_query("I need information about security policies please")
        assert "i" not in result
        assert "information" in result

    def test_punctuation_removed(self):
        """Punctuation tokens should be excluded."""
        doc = _make_spacy_doc([
            {"text": "security", "is_punct": False},
            {"text": ",", "is_punct": True},
            {"text": "policy", "is_punct": False},
        ])
        mock_nlp = Mock(return_value=doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            result = preprocess_query("security, policy and important things here are more words")
        assert "," not in result

    def test_empty_result_falls_back_to_original(self):
        """If all tokens are filtered, returns stripped.lower() of original."""
        # All tokens are stop words or pronouns → preprocessed becomes ""
        doc = _make_spacy_doc([
            {"text": "the", "is_stop": True},
            {"text": "a", "is_stop": True},
        ])
        mock_nlp = Mock(return_value=doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            # Long enough to bypass short-query check
            result = preprocess_query("the a the a the a the a the a")
        # Should fall back to the original (stripped + lowercased)
        assert result == "the a the a the a the a the a"

    def test_tokens_output_are_lowercase(self):
        """Tokens should be lowercased in output."""
        doc = _make_spacy_doc([
            {"text": "Security"},
            {"text": "Policy"},
        ])
        mock_nlp = Mock(return_value=doc)
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            result = preprocess_query("Security Policy regulations compliance documents today")
        assert result == result.lower()


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------

class TestExceptionHandling:
    def test_runtime_error_from_model_loading_is_reraised(self):
        """RuntimeError from _load_spacy_model must propagate."""
        with patch("app.utils.query_preprocessor._load_spacy_model",
                   side_effect=RuntimeError("model not found")):
            from app.utils.query_preprocessor import preprocess_query
            with pytest.raises(RuntimeError, match="model not found"):
                # Use a long query to bypass the short-query fast path
                preprocess_query("find all documents related to security policy compliance")

    def test_other_exception_returns_stripped_original(self):
        """Non-RuntimeError exceptions are caught and original query returned."""
        mock_nlp = Mock(side_effect=ValueError("unexpected nlp error"))
        with patch("app.utils.query_preprocessor._load_spacy_model", return_value=mock_nlp):
            from app.utils.query_preprocessor import preprocess_query
            result = preprocess_query("find all documents related to security policy compliance")
        assert result == "find all documents related to security policy compliance"


# ---------------------------------------------------------------------------
# is_spacy_model_available
# ---------------------------------------------------------------------------

class TestIsSpacyModelAvailable:
    def test_returns_true_when_model_is_installed(self):
        mock_spacy_util = Mock()
        mock_spacy_util.is_package.return_value = True
        with patch.dict("sys.modules", {"spacy.util": mock_spacy_util}):
            # Reimport to pick up the patch — using direct call instead
            with patch("spacy.util.is_package", return_value=True):
                from app.utils.query_preprocessor import is_spacy_model_available
                result = is_spacy_model_available()
        assert result is True

    def test_returns_false_when_model_not_installed(self):
        with patch("spacy.util.is_package", return_value=False):
            from app.utils.query_preprocessor import is_spacy_model_available
            result = is_spacy_model_available()
        assert result is False

    def test_returns_false_on_import_error(self):
        with patch("spacy.util.is_package", side_effect=Exception("import error")):
            from app.utils.query_preprocessor import is_spacy_model_available
            result = is_spacy_model_available()
        assert result is False