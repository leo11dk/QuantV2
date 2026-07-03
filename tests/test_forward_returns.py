import pandas as pd
import pytest

from quantv2.labels.forward_returns import add_forward_return_labels


def _sample_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB", "AAA", "BBB", "AAA", "AAA", "BBB", "BBB", "AAA", "BBB"],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-08",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-09",
                ]
            ),
            "close": [50.0, 110.0, 100.0, 55.0, 130.0, 52.0, 121.0, 143.0, 58.0, 60.0, 150.0, 66.0],
            "volume": [1000, 2100, 2000, 1100, 2300, 1050, 2200, 2400, 1150, 1200, 2500, 1300],
        }
    )


def test_add_forward_return_labels_calculates_expected_returns() -> None:
    labeled = add_forward_return_labels(_sample_prices())

    first_aaa = labeled[labeled["ticker"].eq("AAA")].iloc[0]

    assert first_aaa["decision_date"] == pd.Timestamp("2024-01-02")
    assert first_aaa["close_at_decision"] == 100.0
    assert first_aaa["label_date_1d"] == pd.Timestamp("2024-01-03")
    assert first_aaa["close_1d"] == 110.0
    assert first_aaa["forward_return_1d"] == pytest.approx(0.10)
    assert first_aaa["label_date_3d"] == pd.Timestamp("2024-01-05")
    assert first_aaa["close_3d"] == 130.0
    assert first_aaa["forward_return_3d"] == pytest.approx(0.30)
    assert first_aaa["label_date_5d"] == pd.Timestamp("2024-01-09")
    assert first_aaa["close_5d"] == 150.0
    assert first_aaa["forward_return_5d"] == pytest.approx(0.50)


def test_labels_are_computed_independently_per_ticker() -> None:
    labeled = add_forward_return_labels(_sample_prices())

    aaa_first = labeled[labeled["ticker"].eq("AAA")].iloc[0]
    bbb_first = labeled[labeled["ticker"].eq("BBB")].iloc[0]

    assert aaa_first["label_date_1d"] == pd.Timestamp("2024-01-03")
    assert aaa_first["close_1d"] == 110.0
    assert aaa_first["forward_return_1d"] == pytest.approx(0.10)
    assert bbb_first["label_date_1d"] == pd.Timestamp("2024-01-03")
    assert bbb_first["close_1d"] == 52.0
    assert bbb_first["forward_return_1d"] == pytest.approx(0.04)


def test_unsorted_input_is_sorted_by_ticker_and_date() -> None:
    labeled = add_forward_return_labels(_sample_prices())

    assert labeled["ticker"].tolist() == ["AAA"] * 6 + ["BBB"] * 6
    assert labeled.loc[labeled["ticker"].eq("AAA"), "date"].tolist() == list(
        pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        )
    )
    assert labeled.loc[labeled["ticker"].eq("BBB"), "date"].tolist() == list(
        pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        )
    )


def test_trailing_rows_without_future_observations_remain_missing() -> None:
    labeled = add_forward_return_labels(_sample_prices())
    aaa_rows = labeled[labeled["ticker"].eq("AAA")].reset_index(drop=True)

    assert pd.isna(aaa_rows.loc[5, "label_date_1d"])
    assert pd.isna(aaa_rows.loc[5, "close_1d"])
    assert pd.isna(aaa_rows.loc[5, "forward_return_1d"])
    assert pd.isna(aaa_rows.loc[3, "label_date_3d"])
    assert pd.isna(aaa_rows.loc[3, "close_3d"])
    assert pd.isna(aaa_rows.loc[3, "forward_return_3d"])
    assert pd.isna(aaa_rows.loc[1, "label_date_5d"])
    assert pd.isna(aaa_rows.loc[1, "close_5d"])
    assert pd.isna(aaa_rows.loc[1, "forward_return_5d"])


def test_original_input_dataframe_is_not_mutated() -> None:
    prices = _sample_prices()
    original = prices.copy(deep=True)

    add_forward_return_labels(prices)

    pd.testing.assert_frame_equal(prices, original)
    assert "forward_return_1d" not in prices.columns
    assert "decision_date" not in prices.columns


@pytest.mark.parametrize("missing_column", ["ticker", "date", "close"])
def test_missing_required_columns_raise_clear_value_error(missing_column: str) -> None:
    prices = _sample_prices().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=f"missing required columns: {missing_column}"):
        add_forward_return_labels(prices)


def test_decision_date_equals_row_date_after_sorting() -> None:
    labeled = add_forward_return_labels(_sample_prices())

    pd.testing.assert_series_equal(
        labeled["decision_date"],
        labeled["date"],
        check_names=False,
    )


def test_label_dates_are_after_decision_date_when_available() -> None:
    labeled = add_forward_return_labels(_sample_prices())

    for horizon in (1, 3, 5):
        label_col = f"label_date_{horizon}d"
        rows_with_label = labeled[label_col].notna()

        assert (
            labeled.loc[rows_with_label, label_col]
            > labeled.loc[rows_with_label, "decision_date"]
        ).all()
