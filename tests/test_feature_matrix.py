import pandas as pd
import pytest

from quantv2.features.feature_matrix import build_feature_matrix
from quantv2.features.price_features import add_price_features
from quantv2.labels.forward_returns import add_forward_return_labels


FEATURE_COLUMNS = [
    "prev_close",
    "gap_pct",
    "prior_5d_return",
    "prior_20d_return",
    "volatility_20d",
]
LABEL_COLUMNS = [
    "label_date_1d",
    "close_1d",
    "forward_return_1d",
    "label_date_3d",
    "close_3d",
    "forward_return_3d",
    "label_date_5d",
    "close_5d",
    "forward_return_5d",
]


def _sorted_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    rows = []

    for ticker, base_close, step, gap in (
        ("AAA", 100.0, 2.0, 0.01),
        ("BBB", 200.0, 3.0, -0.02),
    ):
        closes = [base_close + step * day for day in range(len(dates))]
        for index, date in enumerate(dates):
            previous_close = closes[index - 1] if index > 0 else closes[index]
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": previous_close * (1.0 + gap),
                    "close": closes[index],
                    "volume": 1_000 + index,
                }
            )

    return pd.DataFrame(rows)


def _sample_prices() -> pd.DataFrame:
    return _sorted_prices().iloc[::-1].reset_index(drop=True)


def test_feature_matrix_contains_expected_feature_and_label_columns() -> None:
    matrix = build_feature_matrix(_sample_prices())

    assert matrix.columns.tolist() == [
        "ticker",
        "decision_date",
        *FEATURE_COLUMNS,
        *LABEL_COLUMNS,
    ]


def test_feature_matrix_preserves_ticker_and_decision_date() -> None:
    matrix = build_feature_matrix(_sample_prices())

    assert matrix["ticker"].tolist() == ["AAA"] * 25 + ["BBB"] * 25
    assert matrix.loc[matrix["ticker"].eq("AAA"), "decision_date"].tolist() == list(
        pd.date_range("2024-01-01", periods=25, freq="D")
    )
    assert matrix.loc[matrix["ticker"].eq("BBB"), "decision_date"].tolist() == list(
        pd.date_range("2024-01-01", periods=25, freq="D")
    )


def test_feature_values_match_price_feature_output() -> None:
    prices = _sample_prices()
    matrix = build_feature_matrix(prices)
    featured = add_price_features(prices)

    pd.testing.assert_frame_equal(
        matrix[["ticker", "decision_date", *FEATURE_COLUMNS]],
        featured[["ticker", "decision_date", *FEATURE_COLUMNS]],
    )


def test_label_values_match_forward_return_label_output() -> None:
    prices = _sample_prices()
    matrix = build_feature_matrix(prices)
    labeled = add_forward_return_labels(prices)

    pd.testing.assert_frame_equal(
        matrix[["ticker", "decision_date", *LABEL_COLUMNS]],
        labeled[["ticker", "decision_date", *LABEL_COLUMNS]],
    )


def test_input_dataframe_is_not_mutated() -> None:
    prices = _sample_prices()
    original = prices.copy(deep=True)

    build_feature_matrix(prices)

    pd.testing.assert_frame_equal(prices, original)
    assert "gap_pct" not in prices.columns
    assert "forward_return_1d" not in prices.columns


def test_unsorted_input_is_handled_by_sorting_ticker_and_date() -> None:
    matrix = build_feature_matrix(_sample_prices())

    assert matrix["ticker"].tolist() == ["AAA"] * 25 + ["BBB"] * 25
    assert not matrix["decision_date"].is_monotonic_increasing
    assert matrix.loc[matrix["ticker"].eq("AAA"), "decision_date"].is_monotonic_increasing
    assert matrix.loc[matrix["ticker"].eq("BBB"), "decision_date"].is_monotonic_increasing


def test_multiple_tickers_are_handled_independently() -> None:
    matrix = build_feature_matrix(_sample_prices())
    bbb_rows = matrix[matrix["ticker"].eq("BBB")].reset_index(drop=True)

    assert pd.isna(bbb_rows.loc[0, "prev_close"])
    assert pd.isna(bbb_rows.loc[0, "gap_pct"])
    assert bbb_rows.loc[1, "prev_close"] == 200.0
    assert bbb_rows.loc[0, "forward_return_1d"] == pytest.approx(203.0 / 200.0 - 1.0)
    assert bbb_rows.loc[1, "forward_return_1d"] == pytest.approx(206.0 / 203.0 - 1.0)


def test_missing_early_feature_rows_remain_missing() -> None:
    matrix = build_feature_matrix(_sample_prices())
    aaa_rows = matrix[matrix["ticker"].eq("AAA")].reset_index(drop=True)

    assert pd.isna(aaa_rows.loc[0, "prev_close"])
    assert pd.isna(aaa_rows.loc[0, "gap_pct"])
    assert pd.isna(aaa_rows.loc[4, "prior_5d_return"])
    assert pd.isna(aaa_rows.loc[19, "prior_20d_return"])
    assert pd.isna(aaa_rows.loc[19, "volatility_20d"])


def test_missing_trailing_label_rows_remain_missing() -> None:
    matrix = build_feature_matrix(_sample_prices())
    aaa_rows = matrix[matrix["ticker"].eq("AAA")].reset_index(drop=True)

    assert pd.isna(aaa_rows.loc[24, "label_date_1d"])
    assert pd.isna(aaa_rows.loc[24, "close_1d"])
    assert pd.isna(aaa_rows.loc[24, "forward_return_1d"])
    assert pd.isna(aaa_rows.loc[22, "label_date_3d"])
    assert pd.isna(aaa_rows.loc[22, "close_3d"])
    assert pd.isna(aaa_rows.loc[22, "forward_return_3d"])
    assert pd.isna(aaa_rows.loc[20, "label_date_5d"])
    assert pd.isna(aaa_rows.loc[20, "close_5d"])
    assert pd.isna(aaa_rows.loc[20, "forward_return_5d"])


def test_label_dates_are_after_decision_date_when_available() -> None:
    matrix = build_feature_matrix(_sample_prices())

    for horizon in (1, 3, 5):
        label_date_col = f"label_date_{horizon}d"
        has_label_date = matrix[label_date_col].notna()

        assert (
            matrix.loc[has_label_date, label_date_col]
            > matrix.loc[has_label_date, "decision_date"]
        ).all()


def test_future_prices_do_not_change_features_on_or_before_cutoff() -> None:
    prices = _sample_prices()
    changed_future = prices.copy(deep=True)
    cutoff = pd.Timestamp("2024-01-22")
    future_rows = changed_future["date"] > cutoff

    changed_future.loc[future_rows, "open"] = changed_future.loc[future_rows, "open"] * 10.0
    changed_future.loc[future_rows, "close"] = changed_future.loc[future_rows, "close"] * 10.0

    baseline = build_feature_matrix(prices)
    changed = build_feature_matrix(changed_future)
    compare_columns = ["ticker", "decision_date", *FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(
        baseline.loc[
            baseline["decision_date"] <= cutoff,
            compare_columns,
        ].reset_index(drop=True),
        changed.loc[
            changed["decision_date"] <= cutoff,
            compare_columns,
        ].reset_index(drop=True),
    )
