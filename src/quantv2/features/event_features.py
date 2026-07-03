from __future__ import annotations

import pandas as pd


def attach_event_features(
    matrix: pd.DataFrame,
    events: pd.DataFrame,
    event_cols: tuple[str, ...] = ("event_type", "event_direction", "event_severity"),
    ticker_col: str = "ticker",
    decision_date_col: str = "decision_date",
) -> pd.DataFrame:
    """Attach structured event fields to a research matrix.

    Events are matched exactly by ticker and decision date. The function does
    not infer missing event values, create labels, or create trading signals.
    """
    matrix_key_columns = {ticker_col, decision_date_col}
    missing_matrix_columns = sorted(matrix_key_columns.difference(matrix.columns))
    if missing_matrix_columns:
        missing = ", ".join(missing_matrix_columns)
        raise ValueError(f"matrix is missing required columns: {missing}")

    event_required_columns = {ticker_col, decision_date_col, *event_cols}
    missing_event_columns = sorted(event_required_columns.difference(events.columns))
    if missing_event_columns:
        missing = ", ".join(missing_event_columns)
        raise ValueError(f"events is missing required columns: {missing}")

    duplicate_event_rows = events.duplicated(
        subset=[ticker_col, decision_date_col],
        keep=False,
    )
    if duplicate_event_rows.any():
        raise ValueError(
            f"events contains duplicate rows for {ticker_col} and {decision_date_col}"
        )

    attached = matrix.copy(deep=True)
    attached = attached.sort_values(
        [ticker_col, decision_date_col],
        kind="mergesort",
    )
    attached = attached.reset_index(drop=True)

    event_values = events.copy(deep=True)
    event_values = event_values.sort_values(
        [ticker_col, decision_date_col],
        kind="mergesort",
    )
    event_values = event_values.reset_index(drop=True)
    event_values = event_values[[ticker_col, decision_date_col, *event_cols]]

    attached = attached.merge(
        event_values,
        on=[ticker_col, decision_date_col],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    return attached.copy(deep=True)
