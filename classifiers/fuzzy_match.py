"""Fuzzy string matching classifier using RapidFuzz."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


def classify_fuzzy(
    description: str,
    rules: dict[str, str],
    threshold: float = 85.0,
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify a transaction using fuzzy matching against rule keywords."""
    description_upper = (description or "").upper()

    best_keyword = None
    best_category = default_category
    best_score = 0.0

    for keyword, category in rules.items():
        score = float(fuzz.partial_ratio(description_upper, keyword))
        if score > best_score:
            best_score = score
            best_keyword = keyword
            best_category = category

    if best_score >= threshold:
        return {
            "category": best_category,
            "score": best_score,
            "matched_keyword": best_keyword,
            "method": "Fuzzy matching",
        }

    return {
        "category": default_category,
        "score": best_score,
        "matched_keyword": best_keyword,
        "method": "Fuzzy matching",
    }
