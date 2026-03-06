"""Fuzzy string matching classifier using grouped rules."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from utils.rules import normalize


def classify_fuzzy(
    description: str,
    rules: dict[str, list[str]],
    threshold: float = 85.0,
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify by best fuzzy partial_ratio(keyword, normalized_description)."""
    desc_norm = normalize(description)

    best_score = 0.0
    best_group = default_category
    best_keyword = None

    for group, keywords in rules.items():
        for keyword in keywords:
            score = float(fuzz.partial_ratio(keyword, desc_norm))
            if score > best_score:
                best_score = score
                best_group = group
                best_keyword = keyword

    if best_score >= threshold:
        return {
            "category": best_group,
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
