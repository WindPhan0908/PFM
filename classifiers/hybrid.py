"""Hybrid confidence-aware classifier: weighted rules + fuzzy fallback."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


def classify_hybrid(
    description: str,
    rules: dict[str, str],
    rule_weights: dict[str, float],
    rule_threshold: float = 1.0,
    fuzzy_threshold: float = 85.0,
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify using weighted rule confidence, then fuzzy fallback."""
    description_upper = (description or "").upper()

    # Step 1: compute weighted rule score per category.
    category_scores: dict[str, float] = {}
    matched_keywords: list[str] = []

    for keyword, category in rules.items():
        if keyword in description_upper:
            weight = float(rule_weights.get(keyword, 1.0))
            category_scores[category] = category_scores.get(category, 0.0) + weight
            matched_keywords.append(keyword)

    if category_scores:
        best_rule_category, best_rule_score = max(
            category_scores.items(), key=lambda item: item[1]
        )
        if best_rule_score >= rule_threshold:
            return {
                "category": best_rule_category,
                "score": best_rule_score,
                "matched_keyword": ", ".join(matched_keywords),
                "method": "Hybrid confidence-aware (rule score)",
            }

    # Step 2: fallback to fuzzy matching.
    best_keyword = None
    best_category = default_category
    best_fuzzy_score = 0.0

    for keyword, category in rules.items():
        score = float(fuzz.partial_ratio(description_upper, keyword))
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_keyword = keyword
            best_category = category

    if best_fuzzy_score >= fuzzy_threshold:
        return {
            "category": best_category,
            "score": best_fuzzy_score,
            "matched_keyword": best_keyword,
            "method": "Hybrid confidence-aware (fuzzy fallback)",
        }

    return {
        "category": default_category,
        "score": best_fuzzy_score,
        "matched_keyword": best_keyword,
        "method": "Hybrid confidence-aware (default Other)",
    }
