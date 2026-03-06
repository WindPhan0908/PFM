"""Streamlit app for PFM transaction classification experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from classifiers.fuzzy_match import classify_fuzzy
from classifiers.hybrid import classify_hybrid
from classifiers.rule_based import classify_rule_based
from evaluation.metrics import evaluate_dataframe
from utils.rules import DEFAULT_CATEGORY, RULE_WEIGHTS, RULES

st.set_page_config(page_title="PFM Classifier Lab", layout="wide")
st.title("PFM Transaction Classification Experimental Environment")
st.caption(
    "Research sandbox to compare rule-based, fuzzy matching, and hybrid confidence-aware classifiers."
)

METHODS = ["Rule-based", "Fuzzy matching", "Hybrid confidence-aware"]
REQUIRED_COLUMNS = {"transaction_description", "amount", "category"}


def get_predictor(method: str, fuzzy_threshold: float, rule_threshold: float):
    """Return a callable that predicts category payload from text."""
    if method == "Rule-based":
        return lambda text: classify_rule_based(text, RULES, DEFAULT_CATEGORY)
    if method == "Fuzzy matching":
        return lambda text: classify_fuzzy(
            text, RULES, threshold=fuzzy_threshold, default_category=DEFAULT_CATEGORY
        )
    return lambda text: classify_hybrid(
        text,
        RULES,
        RULE_WEIGHTS,
        rule_threshold=rule_threshold,
        fuzzy_threshold=fuzzy_threshold,
        default_category=DEFAULT_CATEGORY,
    )


def validate_dataset_columns(data: pd.DataFrame) -> list[str]:
    """Check whether uploaded data has all required columns."""
    return sorted(REQUIRED_COLUMNS - set(data.columns))


def render_prediction(result: dict) -> None:
    """Render a standard prediction payload to Streamlit."""
    st.subheader("Prediction result")
    st.write(f"**Predicted category:** {result['category']}")
    st.write(f"**Score:** {result['score']:.2f}")
    st.write(f"**Matched keyword(s):** {result.get('matched_keyword')}")
    st.write(f"**Method used:** {result['method']}")


def render_confusion_with_normalized_view(confusion_df: pd.DataFrame) -> None:
    """Show confusion matrix and a normalized row-wise matrix for easier comparison."""
    st.dataframe(confusion_df, use_container_width=True)

    matrix = confusion_df.to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    normalized_df = pd.DataFrame(
        normalized, index=confusion_df.index, columns=confusion_df.columns
    )

    st.caption("Normalized confusion matrix (by true class)")
    st.dataframe(normalized_df.style.format("{:.2f}"), use_container_width=True)


# -----------------------
# Section 1: manual entry
# -----------------------
st.header("1) Manual transaction classification")
manual_left, manual_right = st.columns([2, 1])

with manual_left:
    input_description = st.text_input(
        "Transaction description", placeholder="e.g., Uber trip to office"
    )

with manual_right:
    manual_method = st.selectbox("Method", METHODS, key="manual_method")

manual_fuzzy_threshold = st.slider(
    "Fuzzy threshold", min_value=50, max_value=100, value=85, step=1, key="manual_fuzzy"
)
manual_rule_threshold = st.slider(
    "Rule score threshold (hybrid)",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1,
    key="manual_rule",
)

if st.button("Classify", type="primary"):
    predictor = get_predictor(
        manual_method,
        fuzzy_threshold=float(manual_fuzzy_threshold),
        rule_threshold=float(manual_rule_threshold),
    )
    render_prediction(predictor(input_description))


# -------------------------
# Section 2: dataset upload
# -------------------------
st.header("2) Dataset experiment")
st.markdown(
    "Upload a CSV with columns: `transaction_description`, `amount`, and `category`."
)

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
experiment_method = st.selectbox(
    "Primary method for confusion matrix", METHODS, key="experiment_method"
)

exp_col_1, exp_col_2 = st.columns(2)
with exp_col_1:
    experiment_fuzzy_threshold = st.slider(
        "Experiment fuzzy threshold",
        min_value=50,
        max_value=100,
        value=85,
        step=1,
        key="experiment_fuzzy",
    )
with exp_col_2:
    experiment_rule_threshold = st.slider(
        "Experiment rule score threshold (hybrid)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="experiment_rule",
    )

if st.button("Run Experiment"):
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
    else:
        data = pd.read_csv(uploaded_file)
        missing = validate_dataset_columns(data)

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.success(f"Loaded {len(data)} rows.")

            method_results: dict[str, dict[str, pd.DataFrame]] = {}
            for method in METHODS:
                predictor = get_predictor(
                    method,
                    fuzzy_threshold=float(experiment_fuzzy_threshold),
                    rule_threshold=float(experiment_rule_threshold),
                )
                metrics_df, confusion_df, predicted_df = evaluate_dataframe(data, predictor)
                method_results[method] = {
                    "metrics": metrics_df,
                    "confusion": confusion_df,
                    "predicted": predicted_df,
                }

            st.subheader(f"Metrics ({experiment_method})")
            st.dataframe(method_results[experiment_method]["metrics"], use_container_width=True)

            st.subheader(f"Confusion matrix ({experiment_method})")
            render_confusion_with_normalized_view(
                method_results[experiment_method]["confusion"]
            )

            comparison_rows: list[dict[str, float | str]] = []
            for method in METHODS:
                row = method_results[method]["metrics"].iloc[0].to_dict()
                comparison_rows.append(
                    {
                        "Method": method,
                        "Accuracy": row["Accuracy"],
                        "Precision (macro)": row["Precision (macro)"],
                        "Recall (macro)": row["Recall (macro)"],
                        "F1-score (macro)": row["F1-score (macro)"],
                    }
                )

            comparison_df = pd.DataFrame(comparison_rows)
            st.subheader("Method comparison")
            st.dataframe(comparison_df, use_container_width=True)
            st.bar_chart(
                comparison_df.set_index("Method")[["F1-score (macro)"]],
                use_container_width=True,
            )
