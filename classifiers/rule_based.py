"""Simple rule-based classifier implementation."""

from __future__ import annotations

from typing import Any


def classify_rule_based(
    description: str,
    rules: dict[str, str],
    default_category: str = "Other",
) -> dict[str, Any]:
    """Classify a transaction using direct keyword matching.

    Args:
        description: Free text transaction description.
        rules: Keyword->category mapping.
        default_category: Label returned when no keyword matches.

    Returns:
        A standardized prediction payload with category and confidence details.
    """
    description_upper = (description or "").upper()

    for keyword, category in rules.items():
        if keyword in description_upper:
            return {
                "category": category,
                "score": 1.0,
                "matched_keyword": keyword,
                "method": "Rule-based",
            }

    return {
        "category": default_category,
        "score": 0.0,
        "matched_keyword": None,
        "method": "Rule-based",
    }
