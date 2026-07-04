from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = {
    "prev_close",
    "gap_pct",
    "prior_5d_return",
    "prior_20d_return",
    "volatility_20d",
}
EVENT_COLUMNS = ("event_type", "event_direction", "event_severity")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def build_data_diagnostics(
    data: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "decision_date",
) -> dict[str, pd.DataFrame]:
    """Build data-quality diagnostics for market or research datasets.

    The diagnostics summarize existing columns only. This function does not
    create labels, create features, train models, create trading signals, or
    create live-trading artifacts.
    """
    if ticker_col not in data.columns:
        raise ValueError(f"data is missing ticker column: {ticker_col}")

    date_column_used = _resolve_date_column(data, date_col)
    parsed_dates = pd.to_datetime(data[date_column_used], errors="coerce")
    if len(data) > 0 and parsed_dates.notna().sum() == 0:
        raise ValueError(f"data contains no usable date values in {date_column_used}")

    return {
        "overview": _build_overview(
            data=data,
            tickers=data[ticker_col],
            dates=parsed_dates,
            date_column_used=date_column_used,
        ),
        "missing_values": _build_missing_values(data),
        "duplicates": _build_duplicates(
            tickers=data[ticker_col],
            dates=parsed_dates,
        ),
        "ticker_coverage": _build_ticker_coverage(
            tickers=data[ticker_col],
            dates=parsed_dates,
        ),
        "feature_missingness": _build_feature_missingness(data),
        "label_missingness": _build_label_missingness(data),
        "event_coverage": _build_event_coverage(data),
        "ohlcv_quality": _build_ohlcv_quality(data),
    }


def _resolve_date_column(data: pd.DataFrame, date_col: str) -> str:
    if date_col in data.columns:
        return date_col
    if "date" in data.columns:
        return "date"
    raise ValueError(f"data is missing usable date column: {date_col}")


def _build_overview(
    data: pd.DataFrame,
    tickers: pd.Series,
    dates: pd.Series,
    date_column_used: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_count": int(len(data)),
                "column_count": int(len(data.columns)),
                "ticker_count": int(tickers.nunique(dropna=True)),
                "start_date": dates.min(),
                "end_date": dates.max(),
                "date_column_used": date_column_used,
            }
        ]
    )


def _build_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    missing_counts = data.isna().sum()
    return pd.DataFrame(
        {
            "column": list(data.columns),
            "missing_count": missing_counts.astype(int).to_numpy(),
            "missing_pct": _pct(missing_counts, len(data)).to_numpy(),
        }
    )


def _build_duplicates(tickers: pd.Series, dates: pd.Series) -> pd.DataFrame:
    keys = pd.DataFrame({"ticker": tickers, "date": dates})
    if keys.empty:
        duplicate_key_count = 0
    else:
        key_counts = keys.groupby(["ticker", "date"], dropna=False).size()
        duplicate_key_count = int(key_counts.gt(1).sum())

    return pd.DataFrame([{"duplicate_key_count": duplicate_key_count}])


def _build_ticker_coverage(tickers: pd.Series, dates: pd.Series) -> pd.DataFrame:
    coverage_input = pd.DataFrame({"ticker": tickers, "date": dates})
    if coverage_input.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "row_count",
                "start_date",
                "end_date",
                "unique_date_count",
            ]
        )

    coverage = (
        coverage_input.groupby("ticker", sort=True, dropna=False)
        .agg(
            row_count=("ticker", "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            unique_date_count=("date", "nunique"),
        )
        .reset_index()
    )
    coverage["row_count"] = coverage["row_count"].astype(int)
    coverage["unique_date_count"] = coverage["unique_date_count"].astype(int)
    return coverage[
        ["ticker", "row_count", "start_date", "end_date", "unique_date_count"]
    ]


def _build_feature_missingness(data: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in data.columns if _is_feature_column(column)]
    if not feature_columns:
        return pd.DataFrame(columns=["feature", "missing_count", "missing_pct"])

    missing_counts = data[feature_columns].isna().sum()
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_count": missing_counts.astype(int).to_numpy(),
            "missing_pct": _pct(missing_counts, len(data)).to_numpy(),
        }
    )


def _is_feature_column(column: str) -> bool:
    return (
        column in FEATURE_COLUMNS
        or column.startswith("prior_")
        or column.endswith("_zscore")
        or column.endswith("_feature")
    )


def _build_label_missingness(data: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        column for column in data.columns if column.startswith("forward_return_")
    ]
    if not label_columns:
        return pd.DataFrame(columns=["label", "missing_count", "missing_pct"])

    missing_counts = data[label_columns].isna().sum()
    return pd.DataFrame(
        {
            "label": label_columns,
            "missing_count": missing_counts.astype(int).to_numpy(),
            "missing_pct": _pct(missing_counts, len(data)).to_numpy(),
        }
    )


def _build_event_coverage(data: pd.DataFrame) -> pd.DataFrame:
    event_columns = [column for column in EVENT_COLUMNS if column in data.columns]
    if not event_columns:
        return pd.DataFrame(
            columns=[
                "event_column",
                "non_missing_count",
                "missing_count",
                "coverage_pct",
            ]
        )

    rows = []
    for column in event_columns:
        missing_count = int(data[column].isna().sum())
        non_missing_count = int(data[column].notna().sum())
        rows.append(
            {
                "event_column": column,
                "non_missing_count": non_missing_count,
                "missing_count": missing_count,
                "coverage_pct": _scalar_pct(non_missing_count, len(data)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "event_column",
            "non_missing_count",
            "missing_count",
            "coverage_pct",
        ],
    )


def _build_ohlcv_quality(data: pd.DataFrame) -> pd.DataFrame:
    if not any(column in data.columns for column in OHLCV_COLUMNS):
        return pd.DataFrame(columns=["check", "violation_count"])

    numeric = {
        column: pd.to_numeric(data[column], errors="coerce")
        for column in OHLCV_COLUMNS
        if column in data.columns
    }
    rows: list[dict[str, object]] = []

    for column in ("open", "high", "low", "close"):
        if column in numeric:
            rows.append(
                {
                    "check": f"{column} <= 0",
                    "violation_count": int(numeric[column].le(0).sum()),
                }
            )

    if "volume" in numeric:
        rows.append(
            {
                "check": "volume < 0",
                "violation_count": int(numeric["volume"].lt(0).sum()),
            }
        )

    relationship_checks = (
        ("high < low", "high", "<", "low"),
        ("high < open", "high", "<", "open"),
        ("high < close", "high", "<", "close"),
        ("low > open", "low", ">", "open"),
        ("low > close", "low", ">", "close"),
    )
    for check_name, left_col, operator, right_col in relationship_checks:
        if left_col in numeric and right_col in numeric:
            violations = (
                numeric[left_col].lt(numeric[right_col])
                if operator == "<"
                else numeric[left_col].gt(numeric[right_col])
            )
            rows.append(
                {
                    "check": check_name,
                    "violation_count": int(violations.sum()),
                }
            )

    return pd.DataFrame(rows, columns=["check", "violation_count"])


def _pct(counts: pd.Series, denominator: int) -> pd.Series:
    if denominator == 0:
        return pd.Series([float("nan")] * len(counts), index=counts.index)
    return counts.astype(float) / denominator * 100.0


def _scalar_pct(count: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(count) / denominator * 100.0
