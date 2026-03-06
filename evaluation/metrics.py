"""Evaluation helpers for classification experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute standard macro classification metrics."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall (macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1-score (macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_confusion(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[pd.DataFrame, list[str]]:
    """Build a confusion matrix DataFrame with sorted labels."""
    labels = sorted(set(y_true).union(set(y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return matrix_df, labels


def evaluate_dataframe(
    df: pd.DataFrame,
    predictor,
    description_col: str = "transaction_description",
    label_col: str = "category",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate classifier callable over a dataframe.

    Returns metrics dataframe, confusion matrix dataframe, and dataframe with predictions.
    """
    working_df = df.copy()
    working_df[description_col] = working_df[description_col].fillna("").astype(str)
    working_df[label_col] = working_df[label_col].fillna("Other").astype(str)

    working_df["predicted_category"] = working_df[description_col].apply(
        lambda text: predictor(text)["category"]
    )

    metrics = compute_metrics(working_df[label_col], working_df["predicted_category"])
    confusion_df, _ = compute_confusion(
        working_df[label_col], working_df["predicted_category"]
    )

    metrics_df = pd.DataFrame([metrics])
    return metrics_df, confusion_df, working_df
