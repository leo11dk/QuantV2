from pathlib import Path

import pandas as pd
import pytest

from quantv2.experiments.run_experiment import (
    run_event_study_experiment,
    run_walk_forward_baseline_experiment,
)


EXPECTED_MARKET_ONLY_KEYS = {
    "market_data",
    "feature_matrix",
    "research_data",
    "event_study_report",
    "splits",
    "predictions",
    "prediction_summary",
    "cost_adjusted_summary",
}
EXPECTED_EVENT_KEYS = {*EXPECTED_MARKET_ONLY_KEYS, "event_data"}
EVENT_COLUMNS = ("event_type", "event_direction", "event_severity")
FORBIDDEN_OUTPUT_COLUMNS = {
    "order",
    "execution",
    "fill",
    "position",
    "pnl",
    "profit",
    "brokerage",
}


def _market_data(num_dates: int = 18) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=num_dates, freq="D")
    rows: list[dict[str, object]] = []

    specs = (
        ("AAA", 100.0, 1.03, 0.02),
        ("BBB", 200.0, 0.98, -0.02),
        ("CCC", 50.0, 1.00, 0.00),
    )
    for ticker, base_close, close_multiplier, gap_rate in specs:
        previous_close: float | None = None
        for index, date in enumerate(dates):
            close = base_close * (close_multiplier**index)
            open_price = (
                close if previous_close is None else previous_close * (1 + gap_rate)
            )
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            rows.append(
                {
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1_000 + index,
                }
            )
            previous_close = close

    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def _event_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["BBB", "AAA", "CCC"],
            "decision_date": ["2024-01-11", "2024-01-10", "2024-01-13"],
            "event_type": ["guidance", "earnings", "filing"],
            "event_direction": ["negative", "positive", "neutral"],
            "event_severity": [2.0, 3.0, 1.0],
        }
    )


def _write_csv(tmp_path: Path, name: str, data: pd.DataFrame) -> Path:
    path = tmp_path / name
    data.to_csv(path, index=False)
    return path


def _write_market_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path, "market_data.csv", _market_data())


def _write_event_csv(tmp_path: Path) -> Path:
    return _write_csv(tmp_path, "event_data.csv", _event_data())


def _run_market_only(tmp_path: Path, **kwargs: object) -> dict[str, pd.DataFrame]:
    return run_walk_forward_baseline_experiment(
        _write_market_csv(tmp_path),
        train_window=8,
        test_window=4,
        step_size=4,
        **kwargs,
    )


def _run_with_events(tmp_path: Path, **kwargs: object) -> dict[str, pd.DataFrame]:
    return run_walk_forward_baseline_experiment(
        _write_market_csv(tmp_path),
        _write_event_csv(tmp_path),
        train_window=8,
        test_window=4,
        step_size=4,
        **kwargs,
    )


def test_full_runner_works_with_market_data_only(tmp_path: Path) -> None:
    result = _run_market_only(tmp_path)

    assert set(result) == EXPECTED_MARKET_ONLY_KEYS
    assert not result["event_study_report"].empty
    assert not result["splits"].empty
    assert not result["predictions"].empty
    assert not result["prediction_summary"].empty
    assert not result["cost_adjusted_summary"].empty
    assert "event_data" not in result


def test_full_runner_works_with_market_data_plus_event_data(tmp_path: Path) -> None:
    result = _run_with_events(tmp_path)

    assert set(result) == EXPECTED_EVENT_KEYS
    assert not result["event_data"].empty
    assert not result["event_study_report"].empty
    assert not result["splits"].empty
    assert not result["predictions"].empty
    assert not result["prediction_summary"].empty
    assert not result["cost_adjusted_summary"].empty


def test_research_data_contains_price_features_and_forward_return_labels(
    tmp_path: Path,
) -> None:
    result = _run_market_only(tmp_path)
    research_data = result["research_data"]

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
        assert column in research_data.columns


def test_research_data_event_columns_depend_on_event_data_path(
    tmp_path: Path,
) -> None:
    market_only = _run_market_only(tmp_path)
    with_events = _run_with_events(tmp_path)

    for column in EVENT_COLUMNS:
        assert column not in market_only["research_data"].columns
        assert column in with_events["research_data"].columns

    event_row = with_events["research_data"][
        with_events["research_data"]["ticker"].eq("AAA")
        & with_events["research_data"]["decision_date"].eq(pd.Timestamp("2024-01-10"))
    ].iloc[0]
    assert event_row["event_type"] == "earnings"
    assert event_row["event_direction"] == "positive"


def test_predictions_include_rule_and_cost_adjusted_columns(tmp_path: Path) -> None:
    result = _run_market_only(tmp_path)
    predictions = result["predictions"]

    assert {
        "rule_score",
        "predicted_direction",
        "trade_allowed",
        "no_trade_reason",
        "estimated_round_trip_cost_rate",
        "signed_forward_return_1d",
        "cost_adjusted_signed_forward_return_1d",
    }.issubset(predictions.columns)
    assert predictions["trade_allowed"].any()
    assert predictions["predicted_direction"].eq("no_trade").any()


def test_custom_prediction_kwargs_affect_predictions(tmp_path: Path) -> None:
    default_result = _run_market_only(tmp_path)
    strict_result = _run_market_only(
        tmp_path,
        prediction_kwargs={"min_score_to_trade": 3},
    )

    assert default_result["predictions"]["trade_allowed"].sum() > 0
    assert strict_result["predictions"]["trade_allowed"].sum() == 0


def test_custom_cost_kwargs_affect_estimated_round_trip_cost_rate(
    tmp_path: Path,
) -> None:
    result = _run_market_only(
        tmp_path,
        cost_kwargs={
            "commission_bps": 2.0,
            "slippage_bps": 1.0,
            "extra_cost_bps": 6.0,
        },
    )

    costs = result["predictions"]["estimated_round_trip_cost_rate"]
    assert costs.nunique() == 1
    assert costs.iloc[0] == pytest.approx(0.0012)


def test_custom_report_groupings_are_respected(tmp_path: Path) -> None:
    result = _run_with_events(
        tmp_path,
        report_groupings=(("ticker", "event_type"),),
        include_overall=False,
    )
    report = result["event_study_report"]

    assert set(report["grouping"]) == {"ticker,event_type"}
    assert set(report["report_section"]) == {"by_ticker_and_event_type"}
    assert "event_direction" not in report.columns


def test_validation_errors_propagate_from_existing_loaders_and_splitter(
    tmp_path: Path,
) -> None:
    invalid_market_path = _write_csv(
        tmp_path,
        "invalid_market_data.csv",
        _market_data().drop(columns=["close"]),
    )
    with pytest.raises(
        ValueError,
        match="market data is missing required columns: close",
    ):
        run_walk_forward_baseline_experiment(
            invalid_market_path,
            train_window=8,
            test_window=4,
        )

    market_path = _write_market_csv(tmp_path)
    invalid_event_path = _write_csv(
        tmp_path,
        "invalid_event_data.csv",
        _event_data().drop(columns=["event_type"]),
    )
    with pytest.raises(
        ValueError,
        match="event data is missing required columns: event_type",
    ):
        run_walk_forward_baseline_experiment(
            market_path,
            invalid_event_path,
            train_window=8,
            test_window=4,
        )

    with pytest.raises(ValueError, match="train_window"):
        run_walk_forward_baseline_experiment(
            market_path,
            train_window=0,
            test_window=4,
        )


def test_runner_does_not_create_forbidden_output_columns(tmp_path: Path) -> None:
    result = _run_with_events(tmp_path)

    for frame in result.values():
        lowered_columns = {column.lower() for column in frame.columns}
        assert FORBIDDEN_OUTPUT_COLUMNS.isdisjoint(lowered_columns)


def test_existing_event_study_runner_behavior_is_preserved(tmp_path: Path) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)

    event_study_result = run_event_study_experiment(market_path, event_path)
    full_result = run_walk_forward_baseline_experiment(
        market_path,
        event_path,
        train_window=8,
        test_window=4,
        step_size=4,
    )

    assert set(event_study_result) == {
        "market_data",
        "event_data",
        "feature_matrix",
        "research_data",
        "report",
    }
    assert "event_study_report" not in event_study_result
    pd.testing.assert_frame_equal(
        full_result["event_study_report"],
        event_study_result["report"],
    )


def test_input_csv_files_are_not_modified(tmp_path: Path) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)
    original_market_text = market_path.read_text()
    original_event_text = event_path.read_text()

    run_walk_forward_baseline_experiment(
        market_path,
        event_path,
        train_window=8,
        test_window=4,
        step_size=4,
    )

    assert market_path.read_text() == original_market_text
    assert event_path.read_text() == original_event_text
