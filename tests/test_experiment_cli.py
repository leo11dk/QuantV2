import json
from pathlib import Path

import pandas as pd
import pytest

from quantv2.experiments.cli import main


CORE_MARKET_ONLY_ARTIFACTS = {
    "market_data.csv",
    "feature_matrix.csv",
    "research_data.csv",
    "event_study_report.csv",
    "splits.csv",
    "predictions.csv",
    "prediction_summary.csv",
    "cost_adjusted_summary.csv",
}
CORE_EVENT_ARTIFACTS = {*CORE_MARKET_ONLY_ARTIFACTS, "event_data.csv"}
DIAGNOSTIC_SUFFIXES = (
    "overview",
    "missing_values",
    "duplicates",
    "ticker_coverage",
    "feature_missingness",
    "label_missingness",
    "event_coverage",
    "ohlcv_quality",
)
MARKET_DIAGNOSTIC_ARTIFACTS = {
    f"market_diagnostics_{suffix}.csv" for suffix in DIAGNOSTIC_SUFFIXES
}
RESEARCH_DIAGNOSTIC_ARTIFACTS = {
    f"research_diagnostics_{suffix}.csv" for suffix in DIAGNOSTIC_SUFFIXES
}
EVENT_DIAGNOSTIC_ARTIFACTS = {
    f"event_diagnostics_{suffix}.csv" for suffix in DIAGNOSTIC_SUFFIXES
}
EXPECTED_MARKET_ONLY_ARTIFACTS = {
    *CORE_MARKET_ONLY_ARTIFACTS,
    *MARKET_DIAGNOSTIC_ARTIFACTS,
    *RESEARCH_DIAGNOSTIC_ARTIFACTS,
}
EXPECTED_EVENT_ARTIFACTS = {
    *CORE_EVENT_ARTIFACTS,
    *MARKET_DIAGNOSTIC_ARTIFACTS,
    *RESEARCH_DIAGNOSTIC_ARTIFACTS,
    *EVENT_DIAGNOSTIC_ARTIFACTS,
}
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


def _base_argv(tmp_path: Path, run_id: str) -> list[str]:
    return [
        "--market-data",
        str(_write_market_csv(tmp_path)),
        "--output-dir",
        str(tmp_path / "experiments"),
        "--run-id",
        run_id,
        "--train-window",
        "8",
        "--test-window",
        "4",
        "--step-size",
        "4",
    ]


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "experiments" / "walk_forward_baseline" / run_id


def _artifact_names(run_dir: Path) -> set[str]:
    return {path.name for path in run_dir.glob("*.csv")}


def _manifest_artifact_names(run_dir: Path) -> set[str]:
    with (run_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    return {artifact["name"] for artifact in manifest["artifacts"]}


def _artifact_keys(artifact_names: set[str]) -> set[str]:
    return {Path(artifact_name).stem for artifact_name in artifact_names}


def test_main_works_with_market_data_only_and_prints_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "market_only"

    assert main(_base_argv(tmp_path, run_id)) == 0

    run_dir = _run_dir(tmp_path, run_id)
    captured = capsys.readouterr()
    assert f"Saved run directory: {run_dir}" in captured.out
    assert "predictions.csv" in captured.out
    assert "event_data.csv" not in captured.out
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert _artifact_names(run_dir) == EXPECTED_MARKET_ONLY_ARTIFACTS
    assert _manifest_artifact_names(run_dir) == _artifact_keys(
        EXPECTED_MARKET_ONLY_ARTIFACTS
    )
    assert MARKET_DIAGNOSTIC_ARTIFACTS.issubset(_artifact_names(run_dir))
    assert RESEARCH_DIAGNOSTIC_ARTIFACTS.issubset(_artifact_names(run_dir))
    assert "event_data.csv" not in _artifact_names(run_dir)
    assert EVENT_DIAGNOSTIC_ARTIFACTS.isdisjoint(_artifact_names(run_dir))


def test_main_works_with_market_data_plus_event_data(tmp_path: Path) -> None:
    run_id = "with_events"
    argv = [
        *_base_argv(tmp_path, run_id),
        "--event-data",
        str(_write_event_csv(tmp_path)),
    ]

    assert main(argv) == 0

    run_dir = _run_dir(tmp_path, run_id)
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert _artifact_names(run_dir) == EXPECTED_EVENT_ARTIFACTS
    assert _manifest_artifact_names(run_dir) == _artifact_keys(EXPECTED_EVENT_ARTIFACTS)
    assert "event_data.csv" in _artifact_names(run_dir)
    assert MARKET_DIAGNOSTIC_ARTIFACTS.issubset(_artifact_names(run_dir))
    assert RESEARCH_DIAGNOSTIC_ARTIFACTS.issubset(_artifact_names(run_dir))
    assert EVENT_DIAGNOSTIC_ARTIFACTS.issubset(_artifact_names(run_dir))


def test_run_id_is_respected(tmp_path: Path) -> None:
    run_id = "custom_run_001"

    assert main(_base_argv(tmp_path, run_id)) == 0

    assert _run_dir(tmp_path, run_id).is_dir()


def test_overwrite_works(tmp_path: Path) -> None:
    run_id = "overwrite_run"
    argv = _base_argv(tmp_path, run_id)

    assert main(argv) == 0
    with pytest.raises(ValueError, match="run directory already exists"):
        main(argv)
    assert main([*argv, "--overwrite"]) == 0


def test_custom_horizons_are_parsed_and_passed_through(tmp_path: Path) -> None:
    run_id = "custom_horizons"

    assert main([*_base_argv(tmp_path, run_id), "--horizons", "1,2"]) == 0

    predictions = pd.read_csv(_run_dir(tmp_path, run_id) / "predictions.csv")
    summary = pd.read_csv(_run_dir(tmp_path, run_id) / "prediction_summary.csv")
    assert "forward_return_2d" in predictions.columns
    assert "cost_adjusted_signed_forward_return_2d" in predictions.columns
    assert "forward_return_3d" not in predictions.columns
    assert set(summary["horizon"]) == {1, 2}


def test_custom_return_windows_are_parsed_and_passed_through(tmp_path: Path) -> None:
    run_id = "custom_return_windows"

    assert main([*_base_argv(tmp_path, run_id), "--return-windows", "5,6"]) == 0

    feature_matrix = pd.read_csv(_run_dir(tmp_path, run_id) / "feature_matrix.csv")
    assert "prior_6d_return" in feature_matrix.columns
    assert "prior_20d_return" not in feature_matrix.columns


def test_custom_prediction_arguments_affect_predictions(tmp_path: Path) -> None:
    default_run_id = "default_predictions"
    strict_run_id = "strict_predictions"

    assert main(_base_argv(tmp_path, default_run_id)) == 0
    assert main(
        [
            *_base_argv(tmp_path, strict_run_id),
            "--gap-threshold",
            "0.99",
            "--momentum-threshold",
            "0.99",
            "--min-score-to-trade",
            "3",
            "--max-volatility",
            "0.99",
        ]
    ) == 0

    default_predictions = pd.read_csv(
        _run_dir(tmp_path, default_run_id) / "predictions.csv"
    )
    strict_predictions = pd.read_csv(
        _run_dir(tmp_path, strict_run_id) / "predictions.csv"
    )
    assert default_predictions["trade_allowed"].sum() > 0
    assert strict_predictions["trade_allowed"].sum() == 0


def test_custom_cost_arguments_affect_estimated_round_trip_cost_rate(
    tmp_path: Path,
) -> None:
    run_id = "custom_costs"

    assert main(
        [
            *_base_argv(tmp_path, run_id),
            "--commission-bps",
            "2.0",
            "--slippage-bps",
            "1.0",
            "--extra-cost-bps",
            "6.0",
        ]
    ) == 0

    predictions = pd.read_csv(_run_dir(tmp_path, run_id) / "predictions.csv")
    costs = predictions["estimated_round_trip_cost_rate"]
    assert costs.nunique() == 1
    assert costs.iloc[0] == pytest.approx(0.0012)


def test_invalid_comma_separated_horizons_raise_system_exit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([*_base_argv(tmp_path, "bad_horizons"), "--horizons", "1,bad"])


def test_invalid_comma_separated_return_windows_raise_system_exit(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        main([*_base_argv(tmp_path, "bad_windows"), "--return-windows", "5,,20"])


def test_missing_required_market_data_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_does_not_create_forbidden_output_columns(tmp_path: Path) -> None:
    run_id = "forbidden_columns"
    argv = [
        *_base_argv(tmp_path, run_id),
        "--event-data",
        str(_write_event_csv(tmp_path)),
    ]

    assert main(argv) == 0

    for csv_path in _run_dir(tmp_path, run_id).glob("*.csv"):
        saved_frame = pd.read_csv(csv_path)
        lowered_columns = {column.lower() for column in saved_frame.columns}
        assert FORBIDDEN_OUTPUT_COLUMNS.isdisjoint(lowered_columns)
