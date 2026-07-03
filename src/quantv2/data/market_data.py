from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_market_data(
    data: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Validate daily OHLCV market data for research use.

    The returned frame is a cleaned copy sorted by ticker and date. This
    function does not create labels, create features, fill missing values, or
    fetch external data.
    """
    required_columns = {
        ticker_col,
        date_col,
        open_col,
        high_col,
        low_col,
        close_col,
        volume_col,
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"market data is missing required columns: {missing}")

    validated = data.copy(deep=True)

    ticker_is_missing = validated[ticker_col].isna() | (
        validated[ticker_col].astype("string").str.strip() == ""
    )
    if ticker_is_missing.any():
        raise ValueError("market data contains missing ticker values")

    validated[date_col] = pd.to_datetime(validated[date_col], errors="coerce")
    if validated[date_col].isna().any():
        raise ValueError("market data contains missing or invalid date values")

    if validated.duplicated(subset=[ticker_col, date_col]).any():
        raise ValueError("market data contains duplicate rows for ticker and date")

    ohlc_columns = [open_col, high_col, low_col, close_col]
    numeric_columns = ohlc_columns + [volume_col]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    missing_ohlc_columns = [
        column for column in ohlc_columns if validated[column].isna().any()
    ]
    if missing_ohlc_columns:
        missing = ", ".join(missing_ohlc_columns)
        raise ValueError(f"market data contains missing OHLC values: {missing}")

    if validated[volume_col].isna().any():
        raise ValueError("market data contains missing or invalid volume values")

    non_positive_ohlc_columns = [
        column for column in ohlc_columns if (validated[column] <= 0).any()
    ]
    if non_positive_ohlc_columns:
        invalid = ", ".join(non_positive_ohlc_columns)
        raise ValueError(f"market data contains non-positive OHLC values: {invalid}")

    if (validated[volume_col] < 0).any():
        raise ValueError("market data contains negative volume values")

    invalid_ohlc = (
        (validated[high_col] < validated[low_col])
        | (validated[high_col] < validated[open_col])
        | (validated[high_col] < validated[close_col])
        | (validated[low_col] > validated[open_col])
        | (validated[low_col] > validated[close_col])
    )
    if invalid_ohlc.any():
        raise ValueError("market data contains invalid OHLC relationships")

    return validated.sort_values([ticker_col, date_col], kind="mergesort").reset_index(
        drop=True
    )


def load_market_data_csv(
    path: str | Path,
    ticker_col: str = "ticker",
    date_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Load a CSV file and validate it as daily OHLCV market data."""
    data = pd.read_csv(path)
    return validate_market_data(
        data,
        ticker_col=ticker_col,
        date_col=date_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
    )
