import pandas as pd
import pytest

from quantv2.features.price_features import add_price_features


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


def test_add_price_features_calculates_gap_pct() -> None:
    featured = add_price_features(_sample_prices())
    aaa_rows = featured[featured["ticker"].eq("AAA")].reset_index(drop=True)

    assert pd.isna(aaa_rows.loc[0, "prev_close"])
    assert pd.isna(aaa_rows.loc[0, "gap_pct"])
    assert aaa_rows.loc[1, "prev_close"] == 100.0
    assert aaa_rows.loc[1, "gap_pct"] == pytest.approx(0.01)


def test_add_price_features_calculates_prior_returns() -> None:
    featured = add_price_features(_sample_prices())
    aaa_rows = featured[featured["ticker"].eq("AAA")].reset_index(drop=True)

    assert aaa_rows.loc[20, "prior_5d_return"] == pytest.approx(140.0 / 130.0 - 1.0)
    assert aaa_rows.loc[20, "prior_20d_return"] == pytest.approx(140.0 / 100.0 - 1.0)


def test_add_price_features_calculates_historical_volatility() -> None:
    featured = add_price_features(_sample_prices())
    aaa_rows = featured[featured["ticker"].eq("AAA")].reset_index(drop=True)
    closes = pd.Series([100.0 + 2.0 * day for day in range(21)])
    expected_volatility = closes.pct_change().rolling(window=20, min_periods=20).std().iloc[20]

    assert aaa_rows.loc[20, "volatility_20d"] == pytest.approx(expected_volatility)


def test_features_are_computed_independently_per_ticker() -> None:
    featured = add_price_features(_sample_prices())
    bbb_rows = featured[featured["ticker"].eq("BBB")].reset_index(drop=True)

    assert pd.isna(bbb_rows.loc[0, "prev_close"])
    assert pd.isna(bbb_rows.loc[0, "gap_pct"])
    assert bbb_rows.loc[1, "prev_close"] == 200.0
    assert bbb_rows.loc[1, "gap_pct"] == pytest.approx(-0.02)


def test_unsorted_input_is_sorted_by_ticker_and_date() -> None:
    featured = add_price_features(_sample_prices())

    assert featured["ticker"].tolist() == ["AAA"] * 25 + ["BBB"] * 25
    assert featured.loc[featured["ticker"].eq("AAA"), "date"].tolist() == list(
        pd.date_range("2024-01-01", periods=25, freq="D")
    )
    assert featured.loc[featured["ticker"].eq("BBB"), "date"].tolist() == list(
        pd.date_range("2024-01-01", periods=25, freq="D")
    )


def test_early_rows_without_enough_history_remain_missing() -> None:
    featured = add_price_features(_sample_prices())
    aaa_rows = featured[featured["ticker"].eq("AAA")].reset_index(drop=True)

    assert pd.isna(aaa_rows.loc[0, "prev_close"])
    assert pd.isna(aaa_rows.loc[0, "gap_pct"])
    assert pd.isna(aaa_rows.loc[4, "prior_5d_return"])
    assert pd.isna(aaa_rows.loc[19, "prior_20d_return"])
    assert pd.isna(aaa_rows.loc[19, "volatility_20d"])


def test_original_input_dataframe_is_not_mutated() -> None:
    prices = _sample_prices()
    original = prices.copy(deep=True)

    add_price_features(prices)

    pd.testing.assert_frame_equal(prices, original)
    assert "gap_pct" not in prices.columns
    assert "volatility_20d" not in prices.columns


@pytest.mark.parametrize("missing_column", ["ticker", "date", "open", "close"])
def test_missing_required_columns_raise_clear_value_error(missing_column: str) -> None:
    prices = _sample_prices().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=f"missing required columns: {missing_column}"):
        add_price_features(prices)


def test_decision_date_equals_row_date_after_sorting() -> None:
    featured = add_price_features(_sample_prices())

    pd.testing.assert_series_equal(
        featured["decision_date"],
        featured["date"],
        check_names=False,
    )


def test_future_prices_do_not_change_features_on_or_before_decision_date() -> None:
    prices = _sample_prices()
    changed_future = prices.copy(deep=True)
    cutoff = pd.Timestamp("2024-01-22")
    future_rows = changed_future["date"] > cutoff

    changed_future.loc[future_rows, "open"] = changed_future.loc[future_rows, "open"] * 10.0
    changed_future.loc[future_rows, "close"] = changed_future.loc[future_rows, "close"] * 10.0

    baseline = add_price_features(prices)
    changed = add_price_features(changed_future)
    compare_columns = [
        "ticker",
        "date",
        "decision_date",
        "prev_close",
        "gap_pct",
        "prior_5d_return",
        "prior_20d_return",
        "volatility_20d",
    ]

    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"] <= cutoff, compare_columns].reset_index(drop=True),
        changed.loc[changed["date"] <= cutoff, compare_columns].reset_index(drop=True),
    )
