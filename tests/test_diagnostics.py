import pandas as pd
import pytest

from quantv2.data.diagnostics import build_data_diagnostics


EXPECTED_DIAGNOSTIC_KEYS = {
    "overview",
    "missing_values",
    "duplicates",
    "ticker_coverage",
    "feature_missingness",
    "label_missingness",
    "event_coverage",
    "ohlcv_quality",
}


def _market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB", "AAA"],
            "date": [
                "2024-01-03",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
            "open": [0.0, 11.0, 10.0, 50.0, 11.1],
            "high": [54.0, 9.0, 13.0, 52.0, 12.0],
            "low": [-1.0, 10.0, 12.0, 49.0, 10.0],
            "close": [53.0, 11.5, -1.0, 51.0, 11.6],
            "volume": [1_100, -5, 2_000, 1_000, 2_100],
            "source": ["vendor-b", None, "vendor-a1", "vendor-b1", "vendor-a2"],
        }
    )


def _research_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB"],
            "decision_date": [
                "2024-01-03",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
            ],
            "prev_close": [200.0, None, 100.0, 190.0],
            "gap_pct": [None, 0.01, 0.00, -0.01],
            "prior_5d_return": [0.02, None, 0.01, 0.01],
            "prior_20d_return": [None, None, 0.02, 0.02],
            "volatility_20d": [0.10, None, 0.20, 0.30],
            "sector_zscore": [1.2, None, -0.5, 0.0],
            "quality_feature": [None, 5.0, 4.0, 3.0],
            "forward_return_1d": [0.03, None, 0.01, -0.02],
            "forward_return_3d": [None, 0.05, None, 0.01],
            "event_type": ["earnings", None, "filing", "guidance"],
            "event_direction": [None, "positive", "neutral", "negative"],
            "event_severity": [2.0, 3.0, None, 1.0],
        }
    )


def _row_by(table: pd.DataFrame, column: str, value: object) -> pd.Series:
    return table.loc[table[column].eq(value)].iloc[0]


def _empty_diagnostic(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def test_diagnostics_work_on_market_data_with_date_column() -> None:
    diagnostics = build_data_diagnostics(_market_data())

    assert set(diagnostics) == EXPECTED_DIAGNOSTIC_KEYS
    assert diagnostics["overview"].loc[0, "date_column_used"] == "date"
    assert not diagnostics["ohlcv_quality"].empty


def test_diagnostics_work_on_research_data_with_decision_date_column() -> None:
    diagnostics = build_data_diagnostics(_research_data())

    assert set(diagnostics) == EXPECTED_DIAGNOSTIC_KEYS
    assert diagnostics["overview"].loc[0, "date_column_used"] == "decision_date"
    assert not diagnostics["feature_missingness"].empty
    assert not diagnostics["label_missingness"].empty
    assert not diagnostics["event_coverage"].empty


def test_overview_row_contains_expected_values() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    overview = diagnostics["overview"].iloc[0]

    assert overview["row_count"] == 4
    assert overview["column_count"] == 14
    assert overview["ticker_count"] == 2
    assert overview["start_date"] == pd.Timestamp("2024-01-01")
    assert overview["end_date"] == pd.Timestamp("2024-01-03")
    assert overview["date_column_used"] == "decision_date"


def test_missing_values_are_counted_correctly() -> None:
    diagnostics = build_data_diagnostics(_market_data())
    missing_values = diagnostics["missing_values"]
    source_row = _row_by(missing_values, "column", "source")

    assert source_row["missing_count"] == 1
    assert source_row["missing_pct"] == pytest.approx(20.0)


def test_duplicate_ticker_date_rows_are_counted_correctly() -> None:
    diagnostics = build_data_diagnostics(_market_data())

    assert diagnostics["duplicates"].loc[0, "duplicate_key_count"] == 1


def test_ticker_coverage_is_calculated_correctly() -> None:
    diagnostics = build_data_diagnostics(_market_data())
    coverage = diagnostics["ticker_coverage"]
    aaa_row = _row_by(coverage, "ticker", "AAA")

    assert aaa_row["row_count"] == 3
    assert aaa_row["start_date"] == pd.Timestamp("2024-01-01")
    assert aaa_row["end_date"] == pd.Timestamp("2024-01-02")
    assert aaa_row["unique_date_count"] == 2


def test_feature_missingness_is_calculated_correctly() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    feature_missingness = diagnostics["feature_missingness"]

    prev_close = _row_by(feature_missingness, "feature", "prev_close")
    prior_20d = _row_by(feature_missingness, "feature", "prior_20d_return")
    zscore = _row_by(feature_missingness, "feature", "sector_zscore")
    custom_feature = _row_by(feature_missingness, "feature", "quality_feature")

    assert prev_close["missing_count"] == 1
    assert prev_close["missing_pct"] == pytest.approx(25.0)
    assert prior_20d["missing_count"] == 2
    assert prior_20d["missing_pct"] == pytest.approx(50.0)
    assert zscore["missing_count"] == 1
    assert custom_feature["missing_count"] == 1


def test_label_missingness_is_calculated_correctly() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    label_missingness = diagnostics["label_missingness"]
    one_day = _row_by(label_missingness, "label", "forward_return_1d")
    three_day = _row_by(label_missingness, "label", "forward_return_3d")

    assert one_day["missing_count"] == 1
    assert one_day["missing_pct"] == pytest.approx(25.0)
    assert three_day["missing_count"] == 2
    assert three_day["missing_pct"] == pytest.approx(50.0)


def test_event_coverage_is_calculated_correctly() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    event_coverage = diagnostics["event_coverage"]
    event_type = _row_by(event_coverage, "event_column", "event_type")
    event_direction = _row_by(event_coverage, "event_column", "event_direction")
    event_severity = _row_by(event_coverage, "event_column", "event_severity")

    for row in (event_type, event_direction, event_severity):
        assert row["non_missing_count"] == 3
        assert row["missing_count"] == 1
        assert row["coverage_pct"] == pytest.approx(75.0)


def test_ohlcv_quality_violations_are_counted_correctly() -> None:
    diagnostics = build_data_diagnostics(_market_data())
    quality = diagnostics["ohlcv_quality"]
    expected_counts = {
        "open <= 0": 1,
        "high <= 0": 0,
        "low <= 0": 1,
        "close <= 0": 1,
        "volume < 0": 1,
        "high < low": 1,
        "high < open": 1,
        "high < close": 1,
        "low > open": 1,
        "low > close": 1,
    }

    for check, expected_count in expected_counts.items():
        row = _row_by(quality, "check", check)
        assert row["violation_count"] == expected_count


def test_empty_feature_diagnostics_are_returned_when_no_features_exist() -> None:
    diagnostics = build_data_diagnostics(_market_data())

    pd.testing.assert_frame_equal(
        diagnostics["feature_missingness"],
        _empty_diagnostic(["feature", "missing_count", "missing_pct"]),
    )


def test_empty_label_diagnostics_are_returned_when_no_labels_exist() -> None:
    diagnostics = build_data_diagnostics(_market_data())

    pd.testing.assert_frame_equal(
        diagnostics["label_missingness"],
        _empty_diagnostic(["label", "missing_count", "missing_pct"]),
    )


def test_empty_event_diagnostics_are_returned_when_no_event_columns_exist() -> None:
    diagnostics = build_data_diagnostics(_market_data())

    pd.testing.assert_frame_equal(
        diagnostics["event_coverage"],
        _empty_diagnostic(
            [
                "event_column",
                "non_missing_count",
                "missing_count",
                "coverage_pct",
            ]
        ),
    )


def test_empty_ohlcv_diagnostics_are_returned_when_no_ohlcv_columns_exist() -> None:
    diagnostics = build_data_diagnostics(_research_data())

    pd.testing.assert_frame_equal(
        diagnostics["ohlcv_quality"],
        _empty_diagnostic(["check", "violation_count"]),
    )


def test_input_dataframe_is_not_mutated() -> None:
    data = _research_data()
    original = data.copy(deep=True)

    build_data_diagnostics(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_ticker_column_raises_value_error() -> None:
    data = _research_data().drop(columns=["ticker"])

    with pytest.raises(ValueError, match="data is missing ticker column: ticker"):
        build_data_diagnostics(data)


def test_missing_usable_date_column_raises_value_error() -> None:
    data = _research_data().drop(columns=["decision_date"])

    with pytest.raises(ValueError, match="data is missing usable date column"):
        build_data_diagnostics(data)


def test_unparseable_date_column_raises_value_error() -> None:
    data = pd.DataFrame({"ticker": ["AAA"], "decision_date": ["not-a-date"]})

    with pytest.raises(ValueError, match="data contains no usable date values"):
        build_data_diagnostics(data)


def test_function_does_not_create_generated_or_trading_columns() -> None:
    data = _research_data()
    original_columns = set(data.columns)
    diagnostics = build_data_diagnostics(data)
    forbidden_terms = (
        "signal",
        "trade",
        "order",
        "execution",
        "fill",
        "position",
        "pnl",
        "profit",
        "brokerage",
    )

    assert set(data.columns) == original_columns
    assert set(diagnostics["feature_missingness"]["feature"]).issubset(
        original_columns
    )
    assert set(diagnostics["label_missingness"]["label"]).issubset(original_columns)
    for frame in diagnostics.values():
        for column in frame.columns:
            assert not any(term in column.lower() for term in forbidden_terms)


def test_function_handles_unsorted_input() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    overview = diagnostics["overview"].iloc[0]

    assert overview["start_date"] == pd.Timestamp("2024-01-01")
    assert overview["end_date"] == pd.Timestamp("2024-01-03")


def test_function_handles_multiple_tickers() -> None:
    diagnostics = build_data_diagnostics(_research_data())
    coverage = diagnostics["ticker_coverage"]

    assert coverage["ticker"].tolist() == ["AAA", "BBB"]
    assert coverage["row_count"].tolist() == [2, 2]
