from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_event_data(
    data: pd.DataFrame,
    ticker_col: str = "ticker",
    decision_date_col: str = "decision_date",
    event_cols: tuple[str, ...] = (
        "event_type",
        "event_direction",
        "event_severity",
    ),
) -> pd.DataFrame:
    """Validate structured event data for research use.

    The returned frame is a cleaned copy sorted by ticker and decision date.
    This function does not create labels, create price features, create trading
    signals, infer missing events, or fetch external data.
    """
    required_columns = {ticker_col, decision_date_col, *event_cols}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"event data is missing required columns: {missing}")

    validated = data.copy(deep=True)

    ticker_is_missing = validated[ticker_col].isna() | (
        validated[ticker_col].astype("string").str.strip() == ""
    )
    if ticker_is_missing.any():
        raise ValueError("event data contains missing ticker values")

    validated[decision_date_col] = pd.to_datetime(
        validated[decision_date_col],
        errors="coerce",
    )
    if validated[decision_date_col].isna().any():
        raise ValueError("event data contains missing or invalid decision_date values")

    missing_event_columns = []
    for column in event_cols:
        event_field_is_missing = validated[column].isna() | (
            validated[column].astype("string").str.strip() == ""
        )
        if event_field_is_missing.any():
            missing_event_columns.append(column)

    if missing_event_columns:
        missing = ", ".join(missing_event_columns)
        raise ValueError(f"event data contains missing event field values: {missing}")

    if validated.duplicated(subset=[ticker_col, decision_date_col]).any():
        raise ValueError(
            f"event data contains duplicate rows for {ticker_col} and "
            f"{decision_date_col}"
        )

    return validated.sort_values(
        [ticker_col, decision_date_col],
        kind="mergesort",
    ).reset_index(drop=True)


def load_event_data_csv(
    path: str | Path,
    ticker_col: str = "ticker",
    decision_date_col: str = "decision_date",
    event_cols: tuple[str, ...] = (
        "event_type",
        "event_direction",
        "event_severity",
    ),
) -> pd.DataFrame:
    """Load a CSV file and validate it as structured event data."""
    data = pd.read_csv(path)
    return validate_event_data(
        data,
        ticker_col=ticker_col,
        decision_date_col=decision_date_col,
        event_cols=event_cols,
    )
