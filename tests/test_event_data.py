from pathlib import Path

import pandas as pd
import pytest

from quantv2.data.event_data import load_event_data_csv, validate_event_data


def _sample_event_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB"],
            "decision_date": [
                "2024-01-03",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
            ],
            "event_type": ["filing", "guidance", "earnings", "dividend"],
            "event_direction": ["neutral", "negative", "positive", "positive"],
            "event_severity": [2.0, 4.0, 3.0, 1.0],
            "source": ["vendor-b", "vendor-a2", "vendor-a1", "vendor-b1"],
            "source_reliability": [0.80, 0.95, 0.90, 0.85],
            "notes": ["10-Q filed", "cut outlook", "beat estimates", "cash dividend"],
            "event_time": ["16:05:00", "08:30:00", "07:15:00", "12:00:00"],
            "url": [
                "https://example.test/bbb-filing",
                "https://example.test/aaa-guidance",
                "https://example.test/aaa-earnings",
                "https://example.test/bbb-dividend",
            ],
        }
    )


def test_valid_event_data_is_returned_sorted_by_ticker_and_decision_date() -> None:
    validated = validate_event_data(_sample_event_data())

    assert validated["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert validated["decision_date"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    )


def test_decision_date_is_parsed_to_pandas_datetime() -> None:
    validated = validate_event_data(_sample_event_data())

    assert pd.api.types.is_datetime64_any_dtype(validated["decision_date"])


def test_extra_metadata_columns_are_preserved() -> None:
    validated = validate_event_data(_sample_event_data())

    assert "source" in validated.columns
    assert "source_reliability" in validated.columns
    assert "notes" in validated.columns
    assert "event_time" in validated.columns
    assert "url" in validated.columns
    assert validated["source"].tolist() == [
        "vendor-a1",
        "vendor-a2",
        "vendor-b1",
        "vendor-b",
    ]


def test_validate_event_data_does_not_mutate_input_dataframe() -> None:
    data = _sample_event_data()
    original = data.copy(deep=True)

    validate_event_data(data)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "missing_column",
    ["ticker", "decision_date", "event_type", "event_direction", "event_severity"],
)
def test_missing_required_columns_raise_value_error(missing_column: str) -> None:
    data = _sample_event_data().drop(columns=[missing_column])

    with pytest.raises(
        ValueError,
        match=f"event data is missing required columns: {missing_column}",
    ):
        validate_event_data(data)


def test_duplicate_ticker_decision_date_rows_raise_value_error() -> None:
    data = _sample_event_data()
    data.loc[0, "ticker"] = "AAA"
    data.loc[0, "decision_date"] = "2024-01-01"

    with pytest.raises(
        ValueError,
        match="duplicate rows for ticker and decision_date",
    ):
        validate_event_data(data)


def test_missing_ticker_values_raise_value_error() -> None:
    data = _sample_event_data()
    data.loc[0, "ticker"] = None

    with pytest.raises(ValueError, match="missing ticker values"):
        validate_event_data(data)


def test_missing_decision_date_values_raise_value_error() -> None:
    data = _sample_event_data()
    data.loc[0, "decision_date"] = None

    with pytest.raises(ValueError, match="missing or invalid decision_date values"):
        validate_event_data(data)


@pytest.mark.parametrize(
    "event_column",
    ["event_type", "event_direction", "event_severity"],
)
def test_missing_event_field_values_raise_value_error(event_column: str) -> None:
    data = _sample_event_data()
    data.loc[0, event_column] = None

    with pytest.raises(
        ValueError,
        match=f"missing event field values: {event_column}",
    ):
        validate_event_data(data)


@pytest.mark.parametrize(
    "event_column",
    ["event_type", "event_direction", "event_severity"],
)
@pytest.mark.parametrize("empty_value", ["", "   "])
def test_empty_event_field_strings_raise_value_error(
    event_column: str,
    empty_value: str,
) -> None:
    data = _sample_event_data()
    if event_column == "event_severity":
        data[event_column] = data[event_column].astype("object")
    data.loc[0, event_column] = empty_value

    with pytest.raises(
        ValueError,
        match=f"missing event field values: {event_column}",
    ):
        validate_event_data(data)


def test_load_event_data_csv_reads_and_validates_temporary_csv_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "event_data.csv"
    _sample_event_data().to_csv(path, index=False)

    loaded = load_event_data_csv(path)

    assert loaded["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["decision_date"])


def test_functions_do_not_create_label_columns(tmp_path: Path) -> None:
    path = tmp_path / "event_data.csv"
    _sample_event_data().to_csv(path, index=False)

    validated = validate_event_data(_sample_event_data())
    loaded = load_event_data_csv(path)

    for frame in (validated, loaded):
        assert "forward_return_1d" not in frame.columns


def test_functions_do_not_create_price_feature_columns(tmp_path: Path) -> None:
    path = tmp_path / "event_data.csv"
    _sample_event_data().to_csv(path, index=False)

    validated = validate_event_data(_sample_event_data())
    loaded = load_event_data_csv(path)

    for frame in (validated, loaded):
        assert "gap_pct" not in frame.columns
        assert "prior_5d_return" not in frame.columns


def test_functions_do_not_create_signal_or_trade_columns(tmp_path: Path) -> None:
    path = tmp_path / "event_data.csv"
    _sample_event_data().to_csv(path, index=False)

    validated = validate_event_data(_sample_event_data())
    loaded = load_event_data_csv(path)

    for frame in (validated, loaded):
        assert "signal" not in frame.columns
        assert "trade" not in frame.columns
        assert not any("signal" in column for column in frame.columns)
        assert not any("trade" in column for column in frame.columns)


def test_events_for_different_tickers_on_same_date_remain_separate_rows() -> None:
    validated = validate_event_data(_sample_event_data())
    same_date_rows = validated[
        validated["decision_date"].eq(pd.Timestamp("2024-01-02"))
    ]

    assert same_date_rows["ticker"].tolist() == ["AAA", "BBB"]
    assert same_date_rows["event_type"].tolist() == ["guidance", "dividend"]
