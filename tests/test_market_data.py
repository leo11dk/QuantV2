import pandas as pd
import pytest

from quantv2.data.market_data import load_market_data_csv, validate_market_data


def _sample_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB"],
            "date": ["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-02"],
            "open": [52.0, 11.0, 10.0, 50.0],
            "high": [54.0, 12.0, 11.0, 52.0],
            "low": [51.0, 10.0, 9.0, 49.0],
            "close": [53.0, 11.5, 10.5, 51.0],
            "volume": [1_100, 2_100, 2_000, 1_000],
            "source": ["vendor-b", "vendor-a2", "vendor-a1", "vendor-b1"],
        }
    )


def test_valid_market_data_is_returned_sorted_by_ticker_and_date() -> None:
    validated = validate_market_data(_sample_market_data())

    assert validated["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert validated["date"].tolist() == list(
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    )


def test_date_column_is_parsed_to_pandas_datetime() -> None:
    validated = validate_market_data(_sample_market_data())

    assert pd.api.types.is_datetime64_any_dtype(validated["date"])


def test_extra_columns_are_preserved() -> None:
    validated = validate_market_data(_sample_market_data())

    assert "source" in validated.columns
    assert validated["source"].tolist() == [
        "vendor-a1",
        "vendor-a2",
        "vendor-b1",
        "vendor-b",
    ]


def test_validate_market_data_does_not_mutate_input_dataframe() -> None:
    data = _sample_market_data()
    original = data.copy(deep=True)

    validate_market_data(data)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "missing_column",
    ["ticker", "date", "open", "high", "low", "close", "volume"],
)
def test_missing_required_columns_raise_value_error(missing_column: str) -> None:
    data = _sample_market_data().drop(columns=[missing_column])

    with pytest.raises(
        ValueError,
        match=f"market data is missing required columns: {missing_column}",
    ):
        validate_market_data(data)


def test_duplicate_ticker_date_rows_raise_value_error() -> None:
    data = _sample_market_data()
    data.loc[0, "ticker"] = "AAA"
    data.loc[0, "date"] = "2024-01-01"

    with pytest.raises(
        ValueError,
        match="duplicate rows for ticker and date",
    ):
        validate_market_data(data)


def test_missing_ticker_values_raise_value_error() -> None:
    data = _sample_market_data()
    data.loc[0, "ticker"] = None

    with pytest.raises(ValueError, match="missing ticker values"):
        validate_market_data(data)


def test_missing_date_values_raise_value_error() -> None:
    data = _sample_market_data()
    data.loc[0, "date"] = None

    with pytest.raises(ValueError, match="missing or invalid date values"):
        validate_market_data(data)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_missing_ohlc_values_raise_value_error(column: str) -> None:
    data = _sample_market_data()
    data.loc[0, column] = None

    with pytest.raises(ValueError, match="missing OHLC values"):
        validate_market_data(data)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_zero_or_negative_ohlc_values_raise_value_error(
    column: str,
    value: float,
) -> None:
    data = _sample_market_data()
    data.loc[0, column] = value

    with pytest.raises(ValueError, match="non-positive OHLC values"):
        validate_market_data(data)


def test_negative_volume_raises_value_error() -> None:
    data = _sample_market_data()
    data.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="negative volume values"):
        validate_market_data(data)


def test_zero_volume_is_allowed() -> None:
    data = _sample_market_data()
    data.loc[0, "volume"] = 0

    validated = validate_market_data(data)

    assert 0 in validated["volume"].tolist()


@pytest.mark.parametrize(
    "updates",
    [
        {"high": 8.0},
        {"high": 9.5},
        {"low": 10.25},
        {"low": 10.75},
    ],
)
def test_invalid_ohlc_relationships_raise_value_error(
    updates: dict[str, float],
) -> None:
    data = _sample_market_data()
    for column, value in updates.items():
        data.loc[2, column] = value

    with pytest.raises(ValueError, match="invalid OHLC relationships"):
        validate_market_data(data)


def test_load_market_data_csv_reads_and_validates_temporary_csv_file(tmp_path) -> None:
    path = tmp_path / "market_data.csv"
    _sample_market_data().to_csv(path, index=False)

    loaded = load_market_data_csv(path)

    assert loaded["ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_functions_do_not_create_label_columns(tmp_path) -> None:
    path = tmp_path / "market_data.csv"
    _sample_market_data().to_csv(path, index=False)

    validated = validate_market_data(_sample_market_data())
    loaded = load_market_data_csv(path)

    for frame in (validated, loaded):
        assert "forward_return_1d" not in frame.columns


def test_functions_do_not_create_feature_columns(tmp_path) -> None:
    path = tmp_path / "market_data.csv"
    _sample_market_data().to_csv(path, index=False)

    validated = validate_market_data(_sample_market_data())
    loaded = load_market_data_csv(path)

    for frame in (validated, loaded):
        assert "gap_pct" not in frame.columns
        assert "prior_5d_return" not in frame.columns
