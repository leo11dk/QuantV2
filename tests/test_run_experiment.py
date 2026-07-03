from pathlib import Path

import pandas as pd
import pytest

import quantv2.evaluation.reports as reports
import quantv2.experiments.run_experiment as runner
from quantv2.experiments.run_experiment import run_event_study_experiment


FORBIDDEN_RESEARCH_COLUMNS = ("signal", "trade", "position", "pnl", "profit")
REPORT_REQUIRED_COLUMNS = {
    "report_section",
    "grouping",
    "horizon",
    "count",
    "mean_return",
    "hit_rate",
}


def _market_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    rows = []

    for ticker, base_close, step in (("AAA", 100.0, 1.0), ("BBB", 200.0, 2.0)):
        for index, date in enumerate(dates):
            close = base_close + step * index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "open": close - 0.25,
                    "high": close + 0.50,
                    "low": close - 0.75,
                    "close": close,
                    "volume": 1_000 + index,
                }
            )

    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def _event_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "AAA"],
            "decision_date": ["2024-01-03", "2024-01-02", "2024-01-05"],
            "event_type": ["guidance", "earnings", "filing"],
            "event_direction": ["negative", "positive", "neutral"],
            "event_severity": [2.0, 3.0, 1.0],
        }
    )


def _write_csv(tmp_path: Path, name: str, data: pd.DataFrame) -> Path:
    path = tmp_path / name
    data.to_csv(path, index=False)
    return path


def _run_with_market_data_only(tmp_path: Path) -> dict[str, pd.DataFrame]:
    market_path = _write_csv(tmp_path, "market_data.csv", _market_data())
    return run_event_study_experiment(market_path)


def _run_with_event_data(tmp_path: Path) -> dict[str, pd.DataFrame]:
    market_path = _write_csv(tmp_path, "market_data.csv", _market_data())
    event_path = _write_csv(tmp_path, "event_data.csv", _event_data())
    return run_event_study_experiment(market_path, event_path)


def test_runner_works_with_market_data_only(tmp_path: Path) -> None:
    result = _run_with_market_data_only(tmp_path)

    assert set(result) == {"market_data", "feature_matrix", "research_data", "report"}
    assert not result["feature_matrix"].empty
    assert not result["research_data"].empty
    assert not result["report"].empty


def test_runner_works_with_market_data_plus_event_data(tmp_path: Path) -> None:
    result = _run_with_event_data(tmp_path)

    assert set(result) == {
        "market_data",
        "event_data",
        "feature_matrix",
        "research_data",
        "report",
    }
    assert not result["event_data"].empty
    assert not result["research_data"].empty
    assert not result["report"].empty


def test_market_data_is_validated_and_sorted(tmp_path: Path) -> None:
    result = _run_with_market_data_only(tmp_path)
    market_data = result["market_data"]

    assert market_data["ticker"].tolist() == ["AAA"] * 8 + ["BBB"] * 8
    assert market_data.loc[market_data["ticker"].eq("AAA"), "date"].tolist() == list(
        pd.date_range("2024-01-01", periods=8, freq="D")
    )
    assert market_data.loc[market_data["ticker"].eq("BBB"), "date"].tolist() == list(
        pd.date_range("2024-01-01", periods=8, freq="D")
    )
    assert pd.api.types.is_datetime64_any_dtype(market_data["date"])


def test_feature_matrix_contains_price_features_and_forward_return_labels(
    tmp_path: Path,
) -> None:
    result = _run_with_market_data_only(tmp_path)
    feature_matrix = result["feature_matrix"]

    for column in (
        "prev_close",
        "gap_pct",
        "prior_5d_return",
        "prior_20d_return",
        "volatility_20d",
        "label_date_1d",
        "close_1d",
        "forward_return_1d",
        "label_date_3d",
        "close_3d",
        "forward_return_3d",
        "label_date_5d",
        "close_5d",
        "forward_return_5d",
    ):
        assert column in feature_matrix.columns


def test_research_data_includes_event_columns_when_event_data_is_provided(
    tmp_path: Path,
) -> None:
    result = _run_with_event_data(tmp_path)
    research_data = result["research_data"]

    for column in ("event_type", "event_direction", "event_severity"):
        assert column in research_data.columns

    event_row = research_data[
        research_data["ticker"].eq("AAA")
        & research_data["decision_date"].eq(pd.Timestamp("2024-01-02"))
    ].iloc[0]
    assert event_row["event_type"] == "earnings"
    assert event_row["event_direction"] == "positive"


def test_research_data_excludes_event_columns_without_event_data(
    tmp_path: Path,
) -> None:
    result = _run_with_market_data_only(tmp_path)

    for column in ("event_type", "event_direction", "event_severity"):
        assert column not in result["research_data"].columns


def test_report_is_produced_with_required_columns(tmp_path: Path) -> None:
    result = _run_with_event_data(tmp_path)
    report = result["report"]

    assert REPORT_REQUIRED_COLUMNS.issubset(report.columns)
    assert set(report["horizon"]) == {1, 3, 5}


def test_default_report_groupings_work_with_event_data(tmp_path: Path) -> None:
    result = _run_with_event_data(tmp_path)
    report = result["report"]

    assert set(report["grouping"]) == {
        "overall",
        "ticker",
        "event_type",
        "event_direction",
        "event_type,event_direction",
    }
    assert set(report["report_section"]) == {
        "overall",
        "by_ticker",
        "by_event_type",
        "by_event_direction",
        "by_event_type_and_event_direction",
    }


def test_default_report_groupings_work_without_event_data(tmp_path: Path) -> None:
    result = _run_with_market_data_only(tmp_path)
    report = result["report"]

    assert set(report["grouping"]) == {"overall", "ticker"}
    assert set(report["report_section"]) == {"overall", "by_ticker"}
    assert "event_type" not in report.columns
    assert "event_direction" not in report.columns


def test_custom_report_groupings_are_respected(tmp_path: Path) -> None:
    market_path = _write_csv(tmp_path, "market_data.csv", _market_data())
    event_path = _write_csv(tmp_path, "event_data.csv", _event_data())

    result = run_event_study_experiment(
        market_path,
        event_path,
        report_groupings=(("ticker", "event_type"),),
        include_overall=False,
    )

    report = result["report"]
    assert set(report["grouping"]) == {"ticker,event_type"}
    assert set(report["report_section"]) == {"by_ticker_and_event_type"}
    assert "event_direction" not in set(report["grouping"])


def test_invalid_market_data_raises_value_error_through_loader(tmp_path: Path) -> None:
    invalid_market_data = _market_data().drop(columns=["close"])
    market_path = _write_csv(tmp_path, "market_data.csv", invalid_market_data)

    with pytest.raises(ValueError, match="market data is missing required columns: close"):
        run_event_study_experiment(market_path)


def test_invalid_event_data_raises_value_error_through_loader(tmp_path: Path) -> None:
    market_path = _write_csv(tmp_path, "market_data.csv", _market_data())
    invalid_event_data = _event_data().drop(columns=["event_type"])
    event_path = _write_csv(tmp_path, "event_data.csv", invalid_event_data)

    with pytest.raises(ValueError, match="event data is missing required columns: event_type"):
        run_event_study_experiment(market_path, event_path)


def test_runner_does_not_create_signal_trade_position_pnl_or_profit_columns(
    tmp_path: Path,
) -> None:
    result = _run_with_event_data(tmp_path)

    for frame in result.values():
        for column in frame.columns:
            lowered = column.lower()
            assert not any(forbidden in lowered for forbidden in FORBIDDEN_RESEARCH_COLUMNS)


def test_research_data_is_not_an_alias_for_feature_matrix(tmp_path: Path) -> None:
    result = _run_with_market_data_only(tmp_path)

    result["research_data"].loc[0, "prev_close"] = -999.0

    assert result["feature_matrix"].loc[0, "prev_close"] != -999.0


def test_runner_does_not_bypass_existing_validation_or_metric_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_path = _write_csv(tmp_path, "market_data.csv", _market_data())
    event_path = _write_csv(tmp_path, "event_data.csv", _event_data())
    calls: list[object] = []

    real_market_loader = runner.load_market_data_csv
    real_event_loader = runner.load_event_data_csv
    real_build_feature_matrix = runner.build_feature_matrix
    real_attach_event_features = runner.attach_event_features

    def spy_market_loader(path: str | Path) -> pd.DataFrame:
        calls.append(("market_loader", Path(path).name))
        return real_market_loader(path)

    def spy_event_loader(path: str | Path) -> pd.DataFrame:
        calls.append(("event_loader", Path(path).name))
        return real_event_loader(path)

    def spy_build_feature_matrix(
        prices: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 3, 5),
        return_windows: tuple[int, ...] = (5, 20),
        volatility_window: int = 20,
    ) -> pd.DataFrame:
        calls.append(("feature_matrix", horizons, return_windows, volatility_window))
        return real_build_feature_matrix(
            prices,
            horizons=horizons,
            return_windows=return_windows,
            volatility_window=volatility_window,
        )

    def spy_attach_event_features(
        matrix: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        calls.append(("event_features", len(matrix), len(events)))
        return real_attach_event_features(matrix, events)

    def fake_summary(
        data: pd.DataFrame,
        horizons: tuple[int, ...] = (1, 3, 5),
        group_cols: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        calls.append(("metrics", horizons, group_cols))
        summary_values = {
            "horizon": [horizons[0]],
            "count": [1],
            "missing_count": [0],
            "mean_return": [0.0],
            "median_return": [0.0],
            "std_return": [0.0],
            "min_return": [0.0],
            "max_return": [0.0],
            "hit_rate": [0.0],
        }
        if group_cols is None:
            return pd.DataFrame(summary_values)

        group_values = {
            column: [data[column].dropna().iloc[0]]
            for column in group_cols
        }
        group_values.update(summary_values)
        return pd.DataFrame(group_values)

    monkeypatch.setattr(runner, "load_market_data_csv", spy_market_loader)
    monkeypatch.setattr(runner, "load_event_data_csv", spy_event_loader)
    monkeypatch.setattr(runner, "build_feature_matrix", spy_build_feature_matrix)
    monkeypatch.setattr(runner, "attach_event_features", spy_attach_event_features)
    monkeypatch.setattr(reports, "summarize_forward_returns", fake_summary)

    result = run_event_study_experiment(
        market_path,
        event_path,
        horizons=(1,),
        return_windows=(2,),
        volatility_window=2,
        report_groupings=(("ticker",),),
    )

    assert ("market_loader", "market_data.csv") in calls
    assert ("event_loader", "event_data.csv") in calls
    assert ("feature_matrix", (1,), (2,), 2) in calls
    assert ("event_features", 16, 3) in calls
    assert ("metrics", (1,), None) in calls
    assert ("metrics", (1,), ("ticker",)) in calls
    assert result["report"]["mean_return"].tolist() == [0.0, 0.0]
