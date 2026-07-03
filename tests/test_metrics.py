import pandas as pd
import pytest

from quantv2.evaluation.metrics import summarize_forward_returns


def _sample_research_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "event_type": ["earnings", "guidance", "earnings", "guidance", "earnings"],
            "event_direction": ["positive", "negative", "negative", "positive", "positive"],
            "forward_return_1d": [0.10, -0.05, None, 0.00, 0.20],
            "forward_return_3d": [0.30, None, -0.10, 0.05, None],
            "forward_return_5d": [None, None, None, None, None],
            "close": [100.0, 110.0, 200.0, 190.0, 50.0],
            "future_close": [500.0, 10.0, 300.0, 100.0, 80.0],
        }
    )


def test_overall_summary_works_without_group_cols() -> None:
    summary = summarize_forward_returns(_sample_research_data())

    assert summary.columns.tolist() == [
        "horizon",
        "count",
        "missing_count",
        "mean_return",
        "median_return",
        "std_return",
        "min_return",
        "max_return",
        "hit_rate",
    ]
    assert summary["horizon"].tolist() == [1, 3, 5]

    one_day = summary.loc[summary["horizon"].eq(1)].iloc[0]
    assert one_day["count"] == 4
    assert one_day["missing_count"] == 1
    assert one_day["mean_return"] == pytest.approx(0.0625)
    assert one_day["median_return"] == pytest.approx(0.05)
    assert one_day["std_return"] == pytest.approx(
        pd.Series([0.10, -0.05, 0.00, 0.20]).std()
    )
    assert one_day["min_return"] == pytest.approx(-0.05)
    assert one_day["max_return"] == pytest.approx(0.20)
    assert one_day["hit_rate"] == pytest.approx(0.5)


def test_grouped_summary_works_with_one_group_column() -> None:
    summary = summarize_forward_returns(_sample_research_data(), group_cols=("ticker",))

    assert summary.columns.tolist() == [
        "ticker",
        "horizon",
        "count",
        "missing_count",
        "mean_return",
        "median_return",
        "std_return",
        "min_return",
        "max_return",
        "hit_rate",
    ]

    aaa_one_day = summary[
        summary["ticker"].eq("AAA") & summary["horizon"].eq(1)
    ].iloc[0]
    assert aaa_one_day["count"] == 2
    assert aaa_one_day["missing_count"] == 0
    assert aaa_one_day["mean_return"] == pytest.approx(0.025)
    assert aaa_one_day["hit_rate"] == pytest.approx(0.5)


def test_grouped_summary_works_with_multiple_group_columns() -> None:
    summary = summarize_forward_returns(
        _sample_research_data(),
        group_cols=("event_type", "event_direction"),
    )

    assert summary.columns.tolist() == [
        "event_type",
        "event_direction",
        "horizon",
        "count",
        "missing_count",
        "mean_return",
        "median_return",
        "std_return",
        "min_return",
        "max_return",
        "hit_rate",
    ]

    earnings_positive_one_day = summary[
        summary["event_type"].eq("earnings")
        & summary["event_direction"].eq("positive")
        & summary["horizon"].eq(1)
    ].iloc[0]
    assert earnings_positive_one_day["count"] == 2
    assert earnings_positive_one_day["missing_count"] == 0
    assert earnings_positive_one_day["mean_return"] == pytest.approx(0.15)
    assert earnings_positive_one_day["hit_rate"] == pytest.approx(1.0)


def test_missing_forward_returns_are_counted_correctly() -> None:
    summary = summarize_forward_returns(_sample_research_data())

    three_day = summary.loc[summary["horizon"].eq(3)].iloc[0]
    assert three_day["count"] == 3
    assert three_day["missing_count"] == 2


def test_missing_forward_returns_are_excluded_from_statistics() -> None:
    summary = summarize_forward_returns(_sample_research_data())

    three_day = summary.loc[summary["horizon"].eq(3)].iloc[0]
    expected = pd.Series([0.30, -0.10, 0.05])

    assert three_day["mean_return"] == pytest.approx(expected.mean())
    assert three_day["median_return"] == pytest.approx(expected.median())
    assert three_day["std_return"] == pytest.approx(expected.std())
    assert three_day["min_return"] == pytest.approx(expected.min())
    assert three_day["max_return"] == pytest.approx(expected.max())
    assert three_day["hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_is_share_of_non_missing_positive_forward_returns() -> None:
    summary = summarize_forward_returns(_sample_research_data(), horizons=(1,))

    one_day = summary.iloc[0]
    assert one_day["hit_rate"] == pytest.approx(2 / 4)


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_research_data()
    original = data.copy(deep=True)

    summarize_forward_returns(data)

    pd.testing.assert_frame_equal(data, original)


def test_missing_forward_return_columns_raise_clear_value_error() -> None:
    data = _sample_research_data().drop(columns=["forward_return_5d"])

    with pytest.raises(
        ValueError,
        match="missing required forward-return columns: forward_return_5d",
    ):
        summarize_forward_returns(data)


def test_missing_group_columns_raise_clear_value_error() -> None:
    with pytest.raises(ValueError, match="missing group columns: volatility_regime"):
        summarize_forward_returns(
            _sample_research_data(),
            group_cols=("volatility_regime",),
        )


def test_all_missing_label_groups_are_handled_safely() -> None:
    summary = summarize_forward_returns(_sample_research_data(), group_cols=("ticker",))

    ccc_three_day = summary[
        summary["ticker"].eq("CCC") & summary["horizon"].eq(3)
    ].iloc[0]
    assert ccc_three_day["count"] == 0
    assert ccc_three_day["missing_count"] == 1
    assert pd.isna(ccc_three_day["mean_return"])
    assert pd.isna(ccc_three_day["median_return"])
    assert pd.isna(ccc_three_day["std_return"])
    assert pd.isna(ccc_three_day["min_return"])
    assert pd.isna(ccc_three_day["max_return"])
    assert pd.isna(ccc_three_day["hit_rate"])


def test_horizons_appear_as_integers() -> None:
    summary = summarize_forward_returns(_sample_research_data())

    assert summary["horizon"].tolist() == [1, 3, 5]
    assert all(isinstance(horizon, int) for horizon in summary["horizon"].tolist())


def test_function_only_summarizes_existing_forward_return_columns() -> None:
    summary = summarize_forward_returns(_sample_research_data(), horizons=(1,))

    one_day = summary.iloc[0]
    assert one_day["mean_return"] == pytest.approx(0.0625)
    assert "forward_return_10d" not in summary.columns
    assert "label_date_1d" not in summary.columns
