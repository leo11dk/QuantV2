import pandas as pd
import pytest

from quantv2.backtest.walk_forward import WalkForwardSplit, make_walk_forward_splits


def _sample_data(
    num_dates: int = 30,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    date_as_string: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=num_dates, freq="D")
    rows = []

    for date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "decision_date": date.strftime("%Y-%m-%d") if date_as_string else date,
                    "value": len(rows),
                }
            )

    return pd.DataFrame(rows)


def _split_signature(splits: list[WalkForwardSplit]) -> list[tuple[int, pd.Timestamp, pd.Timestamp, list[int], list[int]]]:
    return [
        (
            split.split_id,
            split.train_start,
            split.test_start,
            split.train_indices.tolist(),
            split.test_indices.tolist(),
        )
        for split in splits
    ]


def test_basic_split_creation_uses_expected_windows() -> None:
    data = _sample_data(num_dates=30, tickers=("AAA",))

    splits = make_walk_forward_splits(
        data,
        train_window=10,
        test_window=5,
        step_size=5,
    )

    assert len(splits) == 4
    assert all(isinstance(split, WalkForwardSplit) for split in splits)

    assert splits[0].split_id == 0
    assert splits[0].train_start == pd.Timestamp("2024-01-01")
    assert splits[0].train_end == pd.Timestamp("2024-01-10")
    assert splits[0].test_start == pd.Timestamp("2024-01-11")
    assert splits[0].test_end == pd.Timestamp("2024-01-15")

    assert splits[3].split_id == 3
    assert splits[3].train_start == pd.Timestamp("2024-01-16")
    assert splits[3].train_end == pd.Timestamp("2024-01-25")
    assert splits[3].test_start == pd.Timestamp("2024-01-26")
    assert splits[3].test_end == pd.Timestamp("2024-01-30")


def test_train_dates_are_strictly_before_test_dates() -> None:
    data = _sample_data(num_dates=18)

    splits = make_walk_forward_splits(data, train_window=6, test_window=3)

    for split in splits:
        train_dates = pd.to_datetime(data.loc[split.train_indices, "decision_date"])
        test_dates = pd.to_datetime(data.loc[split.test_indices, "decision_date"])

        assert train_dates.max() < test_dates.min()


def test_train_and_test_indices_do_not_overlap() -> None:
    data = _sample_data(num_dates=18)

    splits = make_walk_forward_splits(data, train_window=6, test_window=3)

    for split in splits:
        assert split.train_indices.intersection(split.test_indices).empty


def test_splits_use_unique_decision_dates_not_row_counts() -> None:
    data = _sample_data(num_dates=8, tickers=("AAA", "BBB"))

    split = make_walk_forward_splits(data, train_window=3, test_window=2)[0]

    assert len(split.train_indices) == 6
    assert len(split.test_indices) == 4
    assert data.loc[split.train_indices, "decision_date"].nunique() == 3
    assert data.loc[split.test_indices, "decision_date"].nunique() == 2


def test_multiple_tickers_on_same_date_remain_included() -> None:
    data = _sample_data(num_dates=6, tickers=("AAA", "BBB", "CCC"))

    split = make_walk_forward_splits(data, train_window=2, test_window=1)[0]

    train = data.loc[split.train_indices]
    test = data.loc[split.test_indices]

    assert set(train.loc[train["decision_date"].eq(pd.Timestamp("2024-01-01")), "ticker"]) == {"AAA", "BBB", "CCC"}
    assert set(train.loc[train["decision_date"].eq(pd.Timestamp("2024-01-02")), "ticker"]) == {"AAA", "BBB", "CCC"}
    assert set(test.loc[test["decision_date"].eq(pd.Timestamp("2024-01-03")), "ticker"]) == {"AAA", "BBB", "CCC"}


def test_step_size_defaults_to_test_window() -> None:
    data = _sample_data(num_dates=20)

    default_splits = make_walk_forward_splits(data, train_window=5, test_window=4)
    explicit_splits = make_walk_forward_splits(data, train_window=5, test_window=4, step_size=4)

    assert _split_signature(default_splits) == _split_signature(explicit_splits)


def test_custom_step_size_works() -> None:
    data = _sample_data(num_dates=20, tickers=("AAA",))

    splits = make_walk_forward_splits(data, train_window=5, test_window=3, step_size=2)

    assert [split.test_start for split in splits] == list(
        pd.to_datetime(
            [
                "2024-01-06",
                "2024-01-08",
                "2024-01-10",
                "2024-01-12",
                "2024-01-14",
                "2024-01-16",
                "2024-01-18",
            ]
        )
    )


def test_min_train_size_is_respected() -> None:
    data = _sample_data(num_dates=10, tickers=("AAA",))

    splits = make_walk_forward_splits(
        data,
        train_window=5,
        test_window=2,
        step_size=2,
        min_train_size=3,
    )

    assert splits[0].train_start == pd.Timestamp("2024-01-01")
    assert splits[0].train_end == pd.Timestamp("2024-01-03")
    assert splits[0].test_start == pd.Timestamp("2024-01-04")
    assert len(splits[0].train_indices) == 3
    assert all(
        data.loc[split.train_indices, "decision_date"].nunique() >= 3
        for split in splits
    )


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_data(num_dates=12)
    original = data.copy(deep=True)

    make_walk_forward_splits(data, train_window=4, test_window=2)

    pd.testing.assert_frame_equal(data, original)


def test_unsorted_input_is_handled_correctly() -> None:
    data = _sample_data(num_dates=12).iloc[::-1].reset_index(drop=True)

    split = make_walk_forward_splits(data, train_window=4, test_window=2)[0]
    train_dates = pd.to_datetime(data.loc[split.train_indices, "decision_date"])
    test_dates = pd.to_datetime(data.loc[split.test_indices, "decision_date"])

    assert set(train_dates.unique()) == set(pd.date_range("2024-01-01", periods=4, freq="D"))
    assert set(test_dates.unique()) == set(pd.date_range("2024-01-05", periods=2, freq="D"))
    assert train_dates.max() < test_dates.min()


def test_date_col_is_parsed_internally_without_mutating_original_dtype() -> None:
    data = _sample_data(num_dates=8, date_as_string=True)
    original_dtype = data["decision_date"].dtype

    split = make_walk_forward_splits(data, train_window=3, test_window=2)[0]

    assert split.train_start == pd.Timestamp("2024-01-01")
    assert split.test_start == pd.Timestamp("2024-01-04")
    assert data["decision_date"].dtype == original_dtype
    assert isinstance(data.loc[0, "decision_date"], str)


def test_missing_date_col_raises_value_error() -> None:
    data = _sample_data(num_dates=8).drop(columns=["decision_date"])

    with pytest.raises(ValueError, match="missing required column: decision_date"):
        make_walk_forward_splits(data, train_window=3, test_window=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_window": 0},
        {"train_window": -1},
        {"test_window": 0},
        {"test_window": -1},
        {"step_size": 0},
        {"step_size": -1},
        {"min_train_size": 0},
        {"min_train_size": -1},
    ],
)
def test_invalid_window_arguments_raise_value_error(kwargs: dict[str, int]) -> None:
    data = _sample_data(num_dates=8)

    with pytest.raises(ValueError):
        make_walk_forward_splits(data, **kwargs)


def test_not_enough_unique_dates_raises_value_error() -> None:
    data = _sample_data(num_dates=4)

    with pytest.raises(ValueError, match="not enough unique dates"):
        make_walk_forward_splits(data, train_window=3, test_window=2)


def test_splitter_ignores_label_date_columns_and_uses_decision_date_only() -> None:
    data = _sample_data(num_dates=10, tickers=("AAA",))
    data["label_date_1d"] = list(pd.date_range("2030-01-01", periods=len(data), freq="D"))[::-1]

    split = make_walk_forward_splits(data, train_window=3, test_window=2)[0]

    assert split.train_start == pd.Timestamp("2024-01-01")
    assert split.train_end == pd.Timestamp("2024-01-03")
    assert split.test_start == pd.Timestamp("2024-01-04")
    assert split.test_end == pd.Timestamp("2024-01-05")


def test_splitter_does_not_create_research_output_columns() -> None:
    data = _sample_data(num_dates=8)
    original_columns = data.columns.copy()
    forbidden_columns = {"label", "feature", "signal", "trade", "pnl", "profit"}

    make_walk_forward_splits(data, train_window=3, test_window=2)

    assert data.columns.equals(original_columns)
    assert forbidden_columns.isdisjoint(data.columns)
