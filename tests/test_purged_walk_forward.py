import pandas as pd
import pytest

from quantv2.backtest.walk_forward import (
    PurgedWalkForwardSplit,
    make_purged_walk_forward_splits,
    make_walk_forward_splits,
)


def _sample_data(
    num_dates: int = 12,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    dates_as_strings: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for decision_date in pd.date_range("2024-01-01", periods=num_dates, freq="D"):
        for ticker in tickers:
            rows.append(
                {
                    "ticker": ticker,
                    "decision_date": (
                        decision_date.strftime("%Y-%m-%d")
                        if dates_as_strings
                        else decision_date
                    ),
                    "label_date_1d": (
                        (decision_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        if dates_as_strings
                        else decision_date + pd.Timedelta(days=1)
                    ),
                    "label_date_3d": decision_date + pd.Timedelta(days=3),
                    "label_date_5d": decision_date + pd.Timedelta(days=5),
                    "forward_return": len(rows) / 100,
                }
            )

    return pd.DataFrame(rows)


def _ordinary_signature(data: pd.DataFrame) -> list[tuple[object, ...]]:
    return [
        (
            split.split_id,
            split.train_start,
            split.train_end,
            split.test_start,
            split.test_end,
            split.train_indices.tolist(),
            split.test_indices.tolist(),
        )
        for split in make_walk_forward_splits(
            data,
            train_window=4,
            test_window=2,
            step_size=2,
        )
    ]


def test_basic_purged_split_creation_works() -> None:
    data = _sample_data(tickers=("AAA",))

    splits = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
        step_size=2,
    )

    assert len(splits) == 4
    assert all(isinstance(split, PurgedWalkForwardSplit) for split in splits)
    assert splits[0].split_id == 0
    assert splits[0].horizon == 1
    assert splits[0].label_date_col == "label_date_1d"
    assert splits[0].train_start == pd.Timestamp("2024-01-01")
    assert splits[0].train_end == pd.Timestamp("2024-01-04")
    assert splits[0].test_start == pd.Timestamp("2024-01-05")
    assert splits[0].test_end == pd.Timestamp("2024-01-06")


def test_existing_walk_forward_behavior_remains_unchanged() -> None:
    data = _sample_data()
    before = _ordinary_signature(data)

    make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
        step_size=2,
    )

    assert _ordinary_signature(data) == before


def test_retained_training_labels_are_strictly_before_test_start() -> None:
    data = _sample_data()

    splits = make_purged_walk_forward_splits(
        data,
        horizon=3,
        train_window=5,
        test_window=2,
    )

    for split in splits:
        retained_label_dates = pd.to_datetime(
            data.loc[split.train_indices, split.label_date_col]
        )
        assert retained_label_dates.notna().all()
        assert retained_label_dates.lt(split.test_start).all()


def test_equal_after_and_missing_label_dates_are_purged_and_counted() -> None:
    data = _sample_data(num_dates=8, tickers=("AAA",))
    data.loc[1, "label_date_1d"] = pd.NaT
    data.loc[2, "label_date_1d"] = pd.Timestamp("2024-01-06")

    split = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )[0]

    assert split.test_start == pd.Timestamp("2024-01-05")
    assert split.train_indices.tolist() == [0]
    assert 1 not in split.train_indices  # Missing label date.
    assert 2 not in split.train_indices  # Label date after test_start.
    assert 3 not in split.train_indices  # Label date equal to test_start.
    assert split.candidate_train_row_count == 4
    assert split.usable_train_row_count == 1
    assert split.purged_train_row_count == 3
    assert (
        split.candidate_train_row_count
        == split.usable_train_row_count + split.purged_train_row_count
    )


def test_test_indices_and_window_metadata_match_ordinary_splits() -> None:
    data = _sample_data()
    ordinary = make_walk_forward_splits(
        data,
        train_window=4,
        test_window=2,
        step_size=2,
    )
    purged = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
        step_size=2,
    )

    ordinary_by_id = {split.split_id: split for split in ordinary}
    for split in purged:
        candidate = ordinary_by_id[split.split_id]
        assert split.test_indices.equals(candidate.test_indices)
        assert split.train_start == candidate.train_start
        assert split.train_end == candidate.train_end
        assert split.test_start == candidate.test_start
        assert split.test_end == candidate.test_end


@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_default_label_date_naming_works_for_multiple_horizons(horizon: int) -> None:
    data = _sample_data()

    split = make_purged_walk_forward_splits(
        data,
        horizon=horizon,
        train_window=6,
        test_window=2,
    )[0]

    assert split.label_date_col == f"label_date_{horizon}d"


def test_custom_label_date_column_works() -> None:
    data = _sample_data()
    data["outcome_known_at"] = pd.to_datetime(data["decision_date"]) + pd.Timedelta(
        days=2
    )

    split = make_purged_walk_forward_splits(
        data,
        horizon=7,
        label_date_col="outcome_known_at",
        train_window=4,
        test_window=2,
    )[0]

    assert split.horizon == 7
    assert split.label_date_col == "outcome_known_at"


def test_multiple_tickers_are_filtered_independently() -> None:
    data = _sample_data(num_dates=8)
    first_test_start = pd.Timestamp("2024-01-05")
    aaa_first_row = data.index[
        data["ticker"].eq("AAA")
        & data["decision_date"].eq(pd.Timestamp("2024-01-01"))
    ][0]
    bbb_first_row = data.index[
        data["ticker"].eq("BBB")
        & data["decision_date"].eq(pd.Timestamp("2024-01-01"))
    ][0]
    data.loc[aaa_first_row, "label_date_1d"] = first_test_start

    split = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )[0]

    assert aaa_first_row not in split.train_indices
    assert bbb_first_row in split.train_indices
    assert set(data.loc[split.train_indices, "ticker"]) == {"AAA", "BBB"}


def test_same_date_ticker_rows_stay_in_the_proper_windows() -> None:
    data = _sample_data(num_dates=8, tickers=("AAA", "BBB", "CCC"))
    ordinary = make_walk_forward_splits(data, train_window=4, test_window=2)[0]
    purged = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )[0]

    retained = data.loc[purged.train_indices]
    tested = data.loc[purged.test_indices]
    assert set(
        retained.loc[
            retained["decision_date"].eq(pd.Timestamp("2024-01-03")), "ticker"
        ]
    ) == {"AAA", "BBB", "CCC"}
    assert set(
        tested.loc[
            tested["decision_date"].eq(pd.Timestamp("2024-01-05")), "ticker"
        ]
    ) == {"AAA", "BBB", "CCC"}
    assert purged.test_indices.equals(ordinary.test_indices)


def test_unsorted_input_is_handled_correctly() -> None:
    data = _sample_data().sample(frac=1, random_state=9)
    ordinary = make_walk_forward_splits(data, train_window=4, test_window=2)
    purged = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )

    assert [split.test_start for split in purged] == [
        split.test_start for split in ordinary
    ]
    for split in purged:
        train_dates = pd.to_datetime(data.loc[split.train_indices, "decision_date"])
        assert train_dates.max() < split.test_start
        assert split.train_indices.tolist() == [
            index
            for index in ordinary[split.split_id].train_indices
            if pd.Timestamp(data.loc[index, "label_date_1d"]) < split.test_start
        ]


def test_input_and_label_columns_are_not_mutated() -> None:
    data = _sample_data(dates_as_strings=True)
    original = data.copy(deep=True)
    decision_dtype = data["decision_date"].dtype
    label_dtype = data["label_date_1d"].dtype

    make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )

    pd.testing.assert_frame_equal(data, original)
    assert data["decision_date"].dtype == decision_dtype
    assert data["label_date_1d"].dtype == label_dtype
    assert isinstance(data.loc[0, "decision_date"], str)
    assert isinstance(data.loc[0, "label_date_1d"], str)


def test_missing_date_column_raises_value_error() -> None:
    data = _sample_data().drop(columns=["decision_date"])

    with pytest.raises(ValueError, match="missing required column: decision_date"):
        make_purged_walk_forward_splits(data, horizon=1)


def test_missing_label_date_column_raises_value_error() -> None:
    data = _sample_data().drop(columns=["label_date_3d"])

    with pytest.raises(ValueError, match="missing required column: label_date_3d"):
        make_purged_walk_forward_splits(data, horizon=3)


@pytest.mark.parametrize("horizon", [0, -1, 1.0, True, "1"])
def test_invalid_horizon_raises_value_error(horizon: object) -> None:
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        make_purged_walk_forward_splits(_sample_data(), horizon=horizon)  # type: ignore[arg-type]


@pytest.mark.parametrize("minimum", [0, -1, 1.0, True, "1"])
def test_invalid_minimum_usable_rows_raises_value_error(minimum: object) -> None:
    with pytest.raises(
        ValueError, match="min_usable_train_rows must be a positive integer"
    ):
        make_purged_walk_forward_splits(
            _sample_data(),
            horizon=1,
            min_usable_train_rows=minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_label_date",
    [pd.Timestamp("2024-01-01"), pd.Timestamp("2023-12-31")],
)
def test_label_date_not_after_decision_date_raises_value_error(
    invalid_label_date: pd.Timestamp,
) -> None:
    data = _sample_data()
    data.loc[0, "label_date_1d"] = invalid_label_date

    with pytest.raises(ValueError, match="must be strictly after decision_date"):
        make_purged_walk_forward_splits(data, horizon=1)


def test_no_remaining_usable_splits_raises_value_error() -> None:
    data = _sample_data()
    data["label_date_1d"] = pd.Timestamp("2030-01-01")

    with pytest.raises(ValueError, match="no usable purged walk-forward splits"):
        make_purged_walk_forward_splits(
            data,
            horizon=1,
            train_window=4,
            test_window=2,
        )


def test_splits_with_too_few_usable_rows_are_excluded() -> None:
    data = _sample_data(num_dates=10, tickers=("AAA",))
    data.loc[0, "label_date_1d"] = pd.NaT

    splits = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
        min_usable_train_rows=3,
    )

    assert [split.split_id for split in splits] == [1, 2]
    assert all(split.usable_train_row_count >= 3 for split in splits)


def test_forward_returns_do_not_affect_split_membership() -> None:
    data = _sample_data()
    changed_returns = data.copy(deep=True)
    changed_returns["forward_return"] = list(
        reversed(changed_returns["forward_return"].tolist())
    )

    original_splits = make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )
    changed_splits = make_purged_walk_forward_splits(
        changed_returns,
        horizon=1,
        train_window=4,
        test_window=2,
    )

    assert [split.train_indices.tolist() for split in original_splits] == [
        split.train_indices.tolist() for split in changed_splits
    ]
    assert [split.test_indices.tolist() for split in original_splits] == [
        split.test_indices.tolist() for split in changed_splits
    ]


def test_splitter_creates_no_research_output_columns() -> None:
    data = _sample_data()
    original_columns = data.columns.copy()
    forbidden_columns = {
        "label",
        "feature",
        "prediction",
        "signal",
        "trade",
        "order",
        "execution",
        "position",
        "pnl",
        "profit",
    }

    make_purged_walk_forward_splits(
        data,
        horizon=1,
        train_window=4,
        test_window=2,
    )

    assert data.columns.equals(original_columns)
    assert forbidden_columns.isdisjoint(data.columns)
