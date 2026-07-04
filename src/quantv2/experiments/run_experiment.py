from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantv2.backtest.simulator import run_walk_forward_baseline_evaluation
from quantv2.data.event_data import load_event_data_csv
from quantv2.data.market_data import load_market_data_csv
from quantv2.evaluation.reports import build_event_study_report
from quantv2.features.event_features import attach_event_features
from quantv2.features.feature_matrix import build_feature_matrix


MARKET_ONLY_REPORT_GROUPINGS = (("ticker",),)
EVENT_REPORT_GROUPINGS = (
    ("ticker",),
    ("event_type",),
    ("event_direction",),
    ("event_type", "event_direction"),
)


def run_event_study_experiment(
    market_data_path: str | Path,
    event_data_path: str | Path | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
    return_windows: tuple[int, ...] = (5, 20),
    volatility_window: int = 20,
    report_groupings: tuple[tuple[str, ...], ...] | None = None,
    include_overall: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the MVP event-study research pipeline from validated CSV inputs.

    The runner only orchestrates existing loaders, feature builders, event
    attachment logic, and reporting. It does not fetch data, create trading
    signals, create trades, or calculate strategy profitability.
    """
    market_data = load_market_data_csv(market_data_path)
    feature_matrix = build_feature_matrix(
        market_data,
        horizons=horizons,
        return_windows=return_windows,
        volatility_window=volatility_window,
    )

    event_data: pd.DataFrame | None = None
    if event_data_path is None:
        research_data = feature_matrix.copy(deep=True)
    else:
        event_data = load_event_data_csv(event_data_path)
        research_data = attach_event_features(feature_matrix, event_data)

    groupings = report_groupings
    if groupings is None:
        groupings = (
            EVENT_REPORT_GROUPINGS
            if event_data_path is not None
            else MARKET_ONLY_REPORT_GROUPINGS
        )

    report = build_event_study_report(
        research_data,
        horizons=horizons,
        groupings=groupings,
        include_overall=include_overall,
    )

    result = {
        "market_data": market_data.copy(deep=True),
        "feature_matrix": feature_matrix.copy(deep=True),
        "research_data": research_data.copy(deep=True),
        "report": report.copy(deep=True),
    }
    if event_data is not None:
        result["event_data"] = event_data.copy(deep=True)

    return result


def run_walk_forward_baseline_experiment(
    market_data_path: str | Path,
    event_data_path: str | Path | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
    return_windows: tuple[int, ...] = (5, 20),
    volatility_window: int = 20,
    report_groupings: tuple[tuple[str, ...], ...] | None = None,
    include_overall: bool = True,
    train_window: int = 252,
    test_window: int = 21,
    step_size: int | None = None,
    min_train_size: int | None = None,
    prediction_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the MVP walk-forward baseline experiment from validated CSV inputs.

    This function is orchestration only. It reuses the event-study runner for
    loading, feature building, optional event attachment, and reporting, then
    evaluates the deterministic rule baseline over walk-forward test windows.
    Cost-adjusted outputs are research return estimates, not realized PnL.
    """
    event_study_result = run_event_study_experiment(
        market_data_path=market_data_path,
        event_data_path=event_data_path,
        horizons=horizons,
        return_windows=return_windows,
        volatility_window=volatility_window,
        report_groupings=report_groupings,
        include_overall=include_overall,
    )

    baseline_result = run_walk_forward_baseline_evaluation(
        data=event_study_result["research_data"],
        horizons=horizons,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        min_train_size=min_train_size,
        prediction_kwargs={} if prediction_kwargs is None else dict(prediction_kwargs),
        cost_kwargs={} if cost_kwargs is None else dict(cost_kwargs),
    )

    result = {
        "market_data": event_study_result["market_data"].copy(deep=True),
        "feature_matrix": event_study_result["feature_matrix"].copy(deep=True),
        "research_data": event_study_result["research_data"].copy(deep=True),
        "event_study_report": event_study_result["report"].copy(deep=True),
        "splits": baseline_result["splits"].copy(deep=True),
        "predictions": baseline_result["predictions"].copy(deep=True),
        "prediction_summary": baseline_result["prediction_summary"].copy(deep=True),
        "cost_adjusted_summary": baseline_result[
            "cost_adjusted_summary"
        ].copy(deep=True),
    }
    if event_data_path is not None:
        result["event_data"] = event_study_result["event_data"].copy(deep=True)

    return result
