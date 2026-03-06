"""Hybrid confidence-aware classifier (weighted rules + fuzzy fallback)."""

from __future__ import annotations

from typing import Any

from classifiers.fuzzy_match import classify_fuzzy
from utils.rules import normalize


def classify_hybrid(
    description: str,
    rules: dict[str, list[str]],
    rule_weights: dict[str, float],
    rule_threshold: float = 1.0,
    fuzzy_threshold: float = 85.0,
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify using weighted token scores, then fuzzy fallback if needed."""
    tokens = normalize(description).split()

    best_group = default_category
    best_score = 0.0
    best_keywords: list[str] = []

    # Step 1: weighted rule scoring per category.
    for group, keywords in rules.items():
        matched = [kw for kw in keywords if kw in tokens]
        score = float(sum(rule_weights.get(kw, 1) for kw in matched))

        if score > best_score:
            best_score = score
            best_group = group
            best_keywords = matched

    # Step 2: if enough rule confidence, return rule prediction.
    if best_score >= rule_threshold:
        return {
            "category": best_group,
            "score": best_score,
            "matched_keyword": best_keywords,
            "method": "Hybrid confidence-aware (rule score)",
        }

    # Step 3: fuzzy fallback.
    fuzzy_result = classify_fuzzy(
        description,
        rules,
        threshold=fuzzy_threshold,
        default_category=default_category,
    )
    return {
        "category": fuzzy_result["category"],
        "score": fuzzy_result["score"],
        "matched_keyword": fuzzy_result["matched_keyword"],
        "method": "Hybrid confidence-aware (fuzzy fallback)",
    }
