import json
from pathlib import Path

import pandas as pd
import pytest

from quantv2.experiments.run_experiment import (
    run_and_save_walk_forward_baseline_experiment,
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


def _read_manifest(run_dir: Path) -> dict:
    with (run_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _run_saved(
    tmp_path: Path,
    event_data_path: Path | None = None,
    **kwargs: object,
) -> tuple[dict[str, pd.DataFrame], Path]:
    return run_and_save_walk_forward_baseline_experiment(
        market_data_path=_write_market_csv(tmp_path),
        event_data_path=event_data_path,
        output_dir=tmp_path / "experiments",
        train_window=8,
        test_window=4,
        step_size=4,
        **kwargs,
    )


def _assert_saved_artifacts(
    run_dir: Path,
    expected_artifacts: set[str],
) -> None:
    manifest = _read_manifest(run_dir)
    artifacts = {artifact["name"]: artifact for artifact in manifest["artifacts"]}

    assert set(artifacts) == expected_artifacts
    for artifact_name, artifact in artifacts.items():
        artifact_path = run_dir / artifact["path"]
        assert artifact_path == run_dir / f"{artifact_name}.csv"
        assert artifact_path.is_file()

        saved_frame = pd.read_csv(artifact_path)
        assert saved_frame.columns.tolist() == artifact["columns"]
        assert len(saved_frame) == artifact["rows"]


def test_saved_runner_works_with_market_data_only(tmp_path: Path) -> None:
    results, run_dir = _run_saved(tmp_path, run_id="market_only")

    assert isinstance(results, dict)
    assert isinstance(run_dir, Path)
    assert set(results) == EXPECTED_MARKET_ONLY_KEYS
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert "event_data" not in results
    assert not (run_dir / "event_data.csv").exists()
    _assert_saved_artifacts(run_dir, EXPECTED_MARKET_ONLY_KEYS)


def test_saved_runner_works_with_market_data_plus_event_data(tmp_path: Path) -> None:
    event_path = _write_event_csv(tmp_path)
    results, run_dir = _run_saved(
        tmp_path,
        event_data_path=event_path,
        run_id="with_events",
    )

    assert isinstance(results, dict)
    assert isinstance(run_dir, Path)
    assert set(results) == EXPECTED_EVENT_KEYS
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "event_data.csv").is_file()
    _assert_saved_artifacts(run_dir, EXPECTED_EVENT_KEYS)


def test_saved_runner_writes_metadata_without_mutating_input(
    tmp_path: Path,
) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)
    metadata = {"source": "synthetic", "notes": ["tiny", "safe"]}
    original_metadata = {"source": "synthetic", "notes": ["tiny", "safe"]}

    _, run_dir = run_and_save_walk_forward_baseline_experiment(
        market_data_path=market_path,
        event_data_path=event_path,
        output_dir=tmp_path / "experiments",
        experiment_name="saved_baseline",
        metadata=metadata,
        run_id="metadata_run",
        train_window=8,
        test_window=4,
        step_size=4,
    )

    manifest = _read_manifest(run_dir)
    assert metadata == original_metadata
    assert manifest["metadata"]["source"] == "synthetic"
    assert manifest["metadata"]["notes"] == ["tiny", "safe"]
    assert manifest["metadata"]["market_data_path"] == str(market_path)
    assert manifest["metadata"]["event_data_path"] == str(event_path)
    assert manifest["metadata"]["horizons"] == [1, 3, 5]
    assert manifest["metadata"]["return_windows"] == [5, 20]
    assert manifest["metadata"]["volatility_window"] == 20
    assert manifest["metadata"]["train_window"] == 8
    assert manifest["metadata"]["test_window"] == 4
    assert manifest["metadata"]["step_size"] == 4
    assert manifest["metadata"]["min_train_size"] is None
    assert manifest["metadata"]["include_overall"] is True


def test_custom_run_id_and_overwrite_behaviors_use_registry(
    tmp_path: Path,
) -> None:
    run_id = "custom_run_001"

    _, first_run_dir = _run_saved(tmp_path, run_id=run_id)
    assert first_run_dir == tmp_path / "experiments" / "walk_forward_baseline" / run_id

    with pytest.raises(ValueError, match="run directory already exists"):
        _run_saved(tmp_path, run_id=run_id)

    _, overwritten_run_dir = _run_saved(
        tmp_path,
        run_id=run_id,
        metadata={"replacement": True},
        overwrite=True,
    )

    assert overwritten_run_dir == first_run_dir
    assert _read_manifest(overwritten_run_dir)["metadata"]["replacement"] is True


def test_custom_prediction_kwargs_affect_saved_predictions(tmp_path: Path) -> None:
    default_results, _ = _run_saved(tmp_path, run_id="default_predictions")
    strict_results, _ = _run_saved(
        tmp_path,
        run_id="strict_predictions",
        prediction_kwargs={"min_score_to_trade": 3},
    )

    assert default_results["predictions"]["trade_allowed"].sum() > 0
    assert strict_results["predictions"]["trade_allowed"].sum() == 0


def test_saved_runner_rejects_label_date_columns_as_prediction_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="label and future outcome columns",
    ):
        _run_saved(
            tmp_path,
            run_id="label_date_prediction_input",
            prediction_kwargs={"gap_col": "label_date_1d"},
        )


def test_custom_cost_kwargs_affect_saved_cost_estimates(tmp_path: Path) -> None:
    results, run_dir = _run_saved(
        tmp_path,
        run_id="custom_costs",
        cost_kwargs={
            "commission_bps": 2.0,
            "slippage_bps": 1.0,
            "extra_cost_bps": 6.0,
        },
    )

    costs = results["predictions"]["estimated_round_trip_cost_rate"]
    saved_costs = pd.read_csv(run_dir / "predictions.csv")[
        "estimated_round_trip_cost_rate"
    ]
    assert costs.nunique() == 1
    assert saved_costs.nunique() == 1
    assert costs.iloc[0] == pytest.approx(0.0012)
    assert saved_costs.iloc[0] == pytest.approx(0.0012)


def test_validation_errors_propagate_from_existing_logic(tmp_path: Path) -> None:
    invalid_market_path = _write_csv(
        tmp_path,
        "invalid_market_data.csv",
        _market_data().drop(columns=["close"]),
    )
    with pytest.raises(
        ValueError,
        match="market data is missing required columns: close",
    ):
        run_and_save_walk_forward_baseline_experiment(
            market_data_path=invalid_market_path,
            output_dir=tmp_path / "experiments",
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
        run_and_save_walk_forward_baseline_experiment(
            market_data_path=market_path,
            event_data_path=invalid_event_path,
            output_dir=tmp_path / "experiments",
            train_window=8,
            test_window=4,
        )

    with pytest.raises(ValueError, match="train_window"):
        run_and_save_walk_forward_baseline_experiment(
            market_data_path=market_path,
            output_dir=tmp_path / "experiments",
            train_window=0,
            test_window=4,
        )


def test_saved_runner_does_not_create_forbidden_output_columns(
    tmp_path: Path,
) -> None:
    event_path = _write_event_csv(tmp_path)
    results, run_dir = _run_saved(
        tmp_path,
        event_data_path=event_path,
        run_id="forbidden_columns",
    )

    for frame in results.values():
        lowered_columns = {column.lower() for column in frame.columns}
        assert FORBIDDEN_OUTPUT_COLUMNS.isdisjoint(lowered_columns)

    for csv_path in run_dir.glob("*.csv"):
        saved_frame = pd.read_csv(csv_path)
        lowered_columns = {column.lower() for column in saved_frame.columns}
        assert FORBIDDEN_OUTPUT_COLUMNS.isdisjoint(lowered_columns)


def test_existing_event_study_experiment_behavior_is_preserved(
    tmp_path: Path,
) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)

    event_study_result = run_event_study_experiment(market_path, event_path)
    saved_results, _ = run_and_save_walk_forward_baseline_experiment(
        market_data_path=market_path,
        event_data_path=event_path,
        output_dir=tmp_path / "experiments",
        run_id="preserve_event_study",
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
        saved_results["event_study_report"],
        event_study_result["report"],
    )


def test_existing_walk_forward_baseline_experiment_behavior_is_preserved(
    tmp_path: Path,
) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)

    direct_results = run_walk_forward_baseline_experiment(
        market_data_path=market_path,
        event_data_path=event_path,
        train_window=8,
        test_window=4,
        step_size=4,
    )
    saved_results, _ = run_and_save_walk_forward_baseline_experiment(
        market_data_path=market_path,
        event_data_path=event_path,
        output_dir=tmp_path / "experiments",
        run_id="preserve_walk_forward",
        train_window=8,
        test_window=4,
        step_size=4,
    )

    assert set(saved_results) == set(direct_results)
    for artifact_name, direct_frame in direct_results.items():
        pd.testing.assert_frame_equal(saved_results[artifact_name], direct_frame)


def test_input_csv_files_are_not_modified(tmp_path: Path) -> None:
    market_path = _write_market_csv(tmp_path)
    event_path = _write_event_csv(tmp_path)
    original_market_text = market_path.read_text()
    original_event_text = event_path.read_text()

    run_and_save_walk_forward_baseline_experiment(
        market_data_path=market_path,
        event_data_path=event_path,
        output_dir=tmp_path / "experiments",
        run_id="input_immutability",
        train_window=8,
        test_window=4,
        step_size=4,
    )

    assert market_path.read_text() == original_market_text
    assert event_path.read_text() == original_event_text
