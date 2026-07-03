import pandas as pd
import pytest

from quantv2.features.event_features import attach_event_features


def _sample_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA", "BBB", "AAA"],
            "decision_date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "prev_close": [201.0, 100.0, 102.0, 200.0, 101.0],
            "gap_pct": [0.01, 0.02, -0.01, 0.03, 0.00],
        }
    )


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "CCC", "BBB"],
            "decision_date": pd.to_datetime(
                [
                    "2024-01-04",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-04",
                ]
            ),
            "event_type": ["guidance", "dividend", "earnings", "earnings", "filing"],
            "event_direction": ["negative", "positive", "positive", "positive", "neutral"],
            "event_severity": [None, 1.0, 3.0, 5.0, 2.0],
            "event_notes": ["cut", "cash", "beat", "other ticker", "future"],
        }
    )


def test_event_columns_are_attached_by_ticker_and_decision_date() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    row = attached[
        attached["ticker"].eq("AAA")
        & attached["decision_date"].eq(pd.Timestamp("2024-01-02"))
    ].iloc[0]

    assert row["event_type"] == "earnings"
    assert row["event_direction"] == "positive"
    assert row["event_severity"] == 3.0


def test_matrix_rows_without_matching_events_remain_present() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    assert len(attached) == len(_sample_matrix())

    row = attached[
        attached["ticker"].eq("AAA")
        & attached["decision_date"].eq(pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert row["prev_close"] == 101.0
    assert pd.isna(row["event_type"])


def test_missing_event_values_remain_nan_and_are_not_filled() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    row = attached[
        attached["ticker"].eq("AAA")
        & attached["decision_date"].eq(pd.Timestamp("2024-01-04"))
    ].iloc[0]

    assert row["event_type"] == "guidance"
    assert pd.isna(row["event_severity"])


def test_events_do_not_leak_across_tickers() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    row = attached[
        attached["ticker"].eq("BBB")
        & attached["decision_date"].eq(pd.Timestamp("2024-01-02"))
    ].iloc[0]

    assert row["event_type"] == "dividend"
    assert row["event_direction"] == "positive"


def test_events_do_not_leak_across_dates() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    row = attached[
        attached["ticker"].eq("AAA")
        & attached["decision_date"].eq(pd.Timestamp("2024-01-03"))
    ].iloc[0]

    assert pd.isna(row["event_type"])
    assert pd.isna(row["event_direction"])
    assert pd.isna(row["event_severity"])


def test_unsorted_matrix_and_event_inputs_are_handled_correctly() -> None:
    attached = attach_event_features(_sample_matrix(), _sample_events())

    assert attached["ticker"].tolist() == ["AAA", "AAA", "AAA", "BBB", "BBB"]
    assert attached.loc[attached["ticker"].eq("AAA"), "decision_date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    assert attached.loc[attached["ticker"].eq("BBB"), "decision_date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    assert attached.columns.tolist() == [
        "ticker",
        "decision_date",
        "prev_close",
        "gap_pct",
        "event_type",
        "event_direction",
        "event_severity",
    ]


def test_duplicate_events_for_same_ticker_and_decision_date_raise_value_error() -> None:
    events = pd.concat(
        [_sample_events(), _sample_events().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate rows for ticker and decision_date",
    ):
        attach_event_features(_sample_matrix(), events)


@pytest.mark.parametrize("missing_column", ["ticker", "decision_date"])
def test_missing_required_matrix_columns_raise_value_error(missing_column: str) -> None:
    matrix = _sample_matrix().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=f"matrix is missing required columns: {missing_column}"):
        attach_event_features(matrix, _sample_events())


@pytest.mark.parametrize(
    "missing_column",
    ["ticker", "decision_date", "event_type", "event_direction", "event_severity"],
)
def test_missing_required_event_columns_raise_value_error(missing_column: str) -> None:
    events = _sample_events().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=f"events is missing required columns: {missing_column}"):
        attach_event_features(_sample_matrix(), events)


def test_original_matrix_input_is_not_mutated() -> None:
    matrix = _sample_matrix()
    original = matrix.copy(deep=True)

    attach_event_features(matrix, _sample_events())

    pd.testing.assert_frame_equal(matrix, original)
    assert "event_type" not in matrix.columns


def test_original_events_input_is_not_mutated() -> None:
    events = _sample_events()
    original = events.copy(deep=True)

    attach_event_features(_sample_matrix(), events)

    pd.testing.assert_frame_equal(events, original)


def test_function_does_not_create_label_or_signal_columns() -> None:
    matrix = _sample_matrix()[["ticker", "decision_date", "prev_close"]]

    attached = attach_event_features(matrix, _sample_events())
    added_columns = set(attached.columns).difference(matrix.columns)

    assert added_columns == {"event_type", "event_direction", "event_severity"}
    assert not any(column.startswith("label_") for column in attached.columns)
    assert not any(column.startswith("forward_return_") for column in attached.columns)
    assert not any("signal" in column for column in attached.columns)


def test_future_event_date_does_not_affect_earlier_matrix_rows() -> None:
    matrix = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "decision_date": [pd.Timestamp("2024-01-03")],
            "prev_close": [101.0],
        }
    )
    events = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "decision_date": [pd.Timestamp("2024-01-04")],
            "event_type": ["guidance"],
            "event_direction": ["negative"],
            "event_severity": [2.0],
        }
    )

    attached = attach_event_features(matrix, events)

    assert pd.isna(attached.loc[0, "event_type"])
    assert pd.isna(attached.loc[0, "event_direction"])
    assert pd.isna(attached.loc[0, "event_severity"])
