from __future__ import annotations

import pandas as pd


def add_forward_return_labels(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    ticker_col: str = "ticker",
    date_col: str = "date",
    price_col: str = "close",
) -> pd.DataFrame:
    """Add close-to-close forward-return labels to daily price data.

    Each input row is treated as a decision row. For a horizon N, the label uses
    the close and date N rows after the decision row within the same ticker.
    Missing future observations are left missing.
    """
    required_columns = {ticker_col, date_col, price_col}
    missing_columns = sorted(required_columns.difference(prices.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"prices is missing required columns: {missing}")

    invalid_horizons = [
        horizon
        for horizon in horizons
        if not isinstance(horizon, int) or horizon <= 0
    ]
    if invalid_horizons:
        raise ValueError("horizons must contain only positive integers")

    labeled = prices.copy(deep=True)
    labeled = labeled.sort_values([ticker_col, date_col], kind="mergesort")
    labeled = labeled.reset_index(drop=True)

    labeled["decision_date"] = labeled[date_col]
    labeled["close_at_decision"] = labeled[price_col]

    grouped = labeled.groupby(ticker_col, sort=False)

    for horizon in horizons:
        suffix = f"{horizon}d"
        future_date = grouped[date_col].shift(-horizon)
        future_close = grouped[price_col].shift(-horizon)

        labeled[f"label_date_{suffix}"] = future_date
        labeled[f"close_{suffix}"] = future_close
        labeled[f"forward_return_{suffix}"] = (
            future_close / labeled["close_at_decision"] - 1.0
        )

    return labeled
