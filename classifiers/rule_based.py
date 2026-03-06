"""Rule-based classifier using grouped token scoring."""

from __future__ import annotations

from typing import Any

from utils.rules import normalize


def classify_rule_based(
    description: str,
    rules: dict[str, list[str]],
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify by counting exact token matches per group and taking max score."""
    tokens = normalize(description).split()

    best_group = default_category
    best_score = 0
    best_keywords: list[str] = []

    for group, keywords in rules.items():
        matched = [kw for kw in keywords if kw in tokens]
        score = len(matched)

        if score > best_score:
            best_score = score
            best_group = group
            best_keywords = matched

    if best_score > 0:
        return {
            "category": best_group,
            "score": float(best_score),
            "matched_keyword": best_keywords,
            "method": "Rule-based",
        }

    return {
        "category": default_category,
        "score": 0.0,
        "matched_keyword": [],
        "method": "Rule-based",
    }
