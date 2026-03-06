"""Rule definitions and helper constants for transaction classification."""

from __future__ import annotations

# Base keyword -> category mappings used by all methods.
RULES: dict[str, str] = {
    "TESCO": "Groceries",
    "SAINSBURY": "Groceries",
    "ALDI": "Groceries",
    "UBER": "Transport",
    "GRAB": "Transport",
    "SHELL": "Transport",
    "NETFLIX": "Entertainment",
    "SPOTIFY": "Entertainment",
    "STEAM": "Entertainment",
    "SALARY": "Income",
    "PAYROLL": "Income",
    "BONUS": "Income",
    "ELECTRICITY": "Utilities",
    "WATER BILL": "Utilities",
    "INTERNET": "Utilities",
    "PHARMACY": "Healthcare",
    "HOSPITAL": "Healthcare",
    "CLINIC": "Healthcare",
}

# Weight per keyword for the hybrid confidence-aware approach.
RULE_WEIGHTS: dict[str, float] = {
    "TESCO": 1.0,
    "SAINSBURY": 1.0,
    "ALDI": 1.0,
    "UBER": 1.0,
    "GRAB": 1.0,
    "SHELL": 0.8,
    "NETFLIX": 1.1,
    "SPOTIFY": 1.0,
    "STEAM": 1.0,
    "SALARY": 1.2,
    "PAYROLL": 1.1,
    "BONUS": 1.0,
    "ELECTRICITY": 0.9,
    "WATER BILL": 0.9,
    "INTERNET": 0.9,
    "PHARMACY": 0.9,
    "HOSPITAL": 1.0,
    "CLINIC": 0.9,
}

DEFAULT_CATEGORY = "Other"
