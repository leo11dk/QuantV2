from __future__ import annotations

import pandas as pd


def add_price_features(
    prices: pd.DataFrame,
    return_windows: tuple[int, ...] = (5, 20),
    volatility_window: int = 20,
    ticker_col: str = "ticker",
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """Add backward-looking price features to daily price data.

    Each row is treated as a decision row. Features are computed independently
    per ticker using only the current row and prior rows for that ticker.
    """
    required_columns = {ticker_col, date_col, open_col, close_col}
    missing_columns = sorted(required_columns.difference(prices.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"prices is missing required columns: {missing}")

    invalid_windows = [
        window
        for window in return_windows
        if not isinstance(window, int) or window <= 0
    ]
    if invalid_windows:
        raise ValueError("return_windows must contain only positive integers")

    if not isinstance(volatility_window, int) or volatility_window <= 0:
        raise ValueError("volatility_window must be a positive integer")

    featured = prices.copy(deep=True)
    featured = featured.sort_values([ticker_col, date_col], kind="mergesort")
    featured = featured.reset_index(drop=True)

    grouped = featured.groupby(ticker_col, sort=False)

    featured["decision_date"] = featured[date_col]
    featured["prev_close"] = grouped[close_col].shift(1)
    featured["gap_pct"] = featured[open_col] / featured["prev_close"] - 1.0

    for window in return_windows:
        prior_close = grouped[close_col].shift(window)
        featured[f"prior_{window}d_return"] = (
            featured[close_col] / prior_close - 1.0
        )

    daily_returns = grouped[close_col].pct_change()
    featured[f"volatility_{volatility_window}d"] = (
        daily_returns.groupby(featured[ticker_col], sort=False)
        .rolling(window=volatility_window, min_periods=volatility_window)
        .std()
        .reset_index(level=0, drop=True)
    )

    return featured
