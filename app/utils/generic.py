"""Generic shared helpers used across services."""
import json
from typing import Any, Optional


def normalize_theme(value: Optional[Any]) -> Optional[str]:
    """Strip leading and trailing whitespace only; None for a missing or blank value"""
    if value is None:
        return None
    # A theme is a unique code from Commons, so inner spacing and case are kept
    # exactly as sent; upload and search filters must see the same string.
    stripped = str(value).strip()
    return stripped or None


def preview_text(value: Optional[Any], limit: int = 50) -> Optional[str]:
    """Return value as text cut to its first `limit` characters (plus "..."); None stays None"""
    if value is None:
        return None
    # Dicts and lists are shown as JSON, so a metadata preview reads like what was sent
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else f"{text[:limit]}..."
