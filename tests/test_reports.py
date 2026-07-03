import pandas as pd
import pytest

import quantv2.evaluation.reports as reports
from quantv2.evaluation.reports import build_event_study_report


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


SUMMARY_COLUMNS = [
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


def test_overall_report_works() -> None:
    report = build_event_study_report(_sample_research_data())

    assert report.columns.tolist() == ["report_section", "grouping", *SUMMARY_COLUMNS]
    assert report["report_section"].tolist() == ["overall", "overall", "overall"]
    assert report["grouping"].tolist() == ["overall", "overall", "overall"]
    assert report["horizon"].tolist() == [1, 3, 5]

    one_day = report.loc[report["horizon"].eq(1)].iloc[0]
    assert one_day["count"] == 4
    assert one_day["missing_count"] == 1
    assert one_day["mean_return"] == pytest.approx(0.0625)
    assert one_day["hit_rate"] == pytest.approx(0.5)


def test_grouped_report_works_for_one_grouping_column() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("ticker",),),
        include_overall=False,
    )

    assert report.columns.tolist() == [
        "report_section",
        "grouping",
        "ticker",
        *SUMMARY_COLUMNS,
    ]
    assert set(report["report_section"]) == {"by_ticker"}
    assert set(report["grouping"]) == {"ticker"}

    aaa_one_day = report[
        report["ticker"].eq("AAA") & report["horizon"].eq(1)
    ].iloc[0]
    assert aaa_one_day["count"] == 2
    assert aaa_one_day["missing_count"] == 0
    assert aaa_one_day["mean_return"] == pytest.approx(0.025)


def test_grouped_report_works_for_multiple_grouping_columns() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("event_type", "event_direction"),),
        include_overall=False,
    )

    assert report.columns.tolist() == [
        "report_section",
        "grouping",
        "event_type",
        "event_direction",
        *SUMMARY_COLUMNS,
    ]
    assert set(report["report_section"]) == {"by_event_type_and_event_direction"}
    assert set(report["grouping"]) == {"event_type,event_direction"}

    row = report[
        report["event_type"].eq("earnings")
        & report["event_direction"].eq("positive")
        & report["horizon"].eq(1)
    ].iloc[0]
    assert row["count"] == 2
    assert row["mean_return"] == pytest.approx(0.15)
    assert row["hit_rate"] == pytest.approx(1.0)


def test_overall_and_grouped_sections_can_be_included_together() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("ticker",), ("event_type",)),
    )

    assert set(report["report_section"]) == {
        "overall",
        "by_ticker",
        "by_event_type",
    }
    assert set(report["grouping"]) == {"overall", "ticker", "event_type"}
    assert report.loc[report["report_section"].eq("overall"), "ticker"].isna().all()
    assert report.loc[report["report_section"].eq("overall"), "event_type"].isna().all()


def test_include_overall_false_works_when_groupings_are_provided() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("event_type",),),
        include_overall=False,
    )

    assert set(report["report_section"]) == {"by_event_type"}
    assert "overall" not in set(report["grouping"])


@pytest.mark.parametrize("groupings", [None, ()])
def test_include_overall_false_with_no_groupings_raises_value_error(
    groupings: tuple[tuple[str, ...], ...] | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="include_overall=False requires at least one grouping",
    ):
        build_event_study_report(
            _sample_research_data(),
            groupings=groupings,
            include_overall=False,
        )


def test_missing_grouping_columns_raise_value_error() -> None:
    with pytest.raises(ValueError, match="missing grouping columns: volatility_regime"):
        build_event_study_report(
            _sample_research_data(),
            groupings=(("volatility_regime",),),
        )


def test_missing_forward_return_columns_raise_value_error_through_metrics_validation() -> None:
    data = _sample_research_data().drop(columns=["forward_return_5d"])

    with pytest.raises(
        ValueError,
        match="missing required forward-return columns: forward_return_5d",
    ):
        build_event_study_report(data)


def test_input_dataframe_is_not_mutated() -> None:
    data = _sample_research_data()
    original = data.copy(deep=True)

    build_event_study_report(data, groupings=(("ticker",),))

    pd.testing.assert_frame_equal(data, original)


def test_missing_labels_remain_missing_and_are_counted_correctly() -> None:
    report = build_event_study_report(_sample_research_data())

    three_day = report.loc[report["horizon"].eq(3)].iloc[0]
    assert three_day["count"] == 3
    assert three_day["missing_count"] == 2
    assert three_day["mean_return"] == pytest.approx(pd.Series([0.30, -0.10, 0.05]).mean())


def test_report_includes_report_section_and_grouping_columns() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("event_type",),),
    )

    assert "report_section" in report.columns
    assert "grouping" in report.columns
    assert "event_type" in report.columns


def test_function_calls_existing_metric_behavior_without_generating_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[str, ...] | None]] = []

    def fake_summary(
        data: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 3, 5),
        group_cols: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        calls.append((horizons, group_cols))
        if group_cols is None:
            return pd.DataFrame(
                {
                    "horizon": [1],
                    "count": [len(data)],
                    "missing_count": [0],
                    "mean_return": [0.0],
                    "median_return": [0.0],
                    "std_return": [0.0],
                    "min_return": [0.0],
                    "max_return": [0.0],
                    "hit_rate": [0.0],
                }
            )

        return pd.DataFrame(
            {
                group_cols[0]: ["AAA"],
                "horizon": [1],
                "count": [1],
                "missing_count": [0],
                "mean_return": [0.0],
                "median_return": [0.0],
                "std_return": [0.0],
                "min_return": [0.0],
                "max_return": [0.0],
                "hit_rate": [0.0],
            }
        )

    monkeypatch.setattr(reports, "summarize_forward_returns", fake_summary)

    data = _sample_research_data().drop(
        columns=["forward_return_1d", "forward_return_3d", "forward_return_5d"]
    )
    report = reports.build_event_study_report(
        data,
        horizons=(1,),
        groupings=(("ticker",),),
    )

    assert calls == [((1,), None), ((1,), ("ticker",))]
    assert "forward_return_1d" not in report.columns
    assert "label_date_1d" not in report.columns


def test_function_does_not_create_signal_trade_or_profit_columns() -> None:
    report = build_event_study_report(
        _sample_research_data(),
        groupings=(("ticker",),),
    )

    assert not any("signal" in column for column in report.columns)
    assert not any("trade" in column for column in report.columns)
    assert not any("profit" in column for column in report.columns)
