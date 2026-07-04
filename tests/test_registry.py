import json
from pathlib import Path

import pandas as pd
import pytest

from quantv2.experiments.registry import save_experiment_results


FORBIDDEN_CREATED_COLUMNS = (
    "order",
    "execution",
    "fill",
    "position",
    "pnl",
    "profit",
    "brokerage",
)


def _read_manifest(run_dir: Path) -> dict:
    with (run_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _sample_results() -> dict[str, pd.DataFrame]:
    return {
        "research_data": pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "score": [0.25, -0.10],
                "decision_date": ["2024-01-02", "2024-01-03"],
            }
        ),
        "summary": pd.DataFrame(
            {
                "metric": ["rows", "mean_score"],
                "value": [2.0, 0.075],
            }
        ),
    }


def test_save_experiment_results_creates_directory_csvs_and_manifest(
    tmp_path: Path,
) -> None:
    results = _sample_results()
    metadata = {"source": "synthetic", "horizons": [1, 3, 5]}

    run_dir = save_experiment_results(
        results=results,
        output_dir=tmp_path / "experiments",
        experiment_name="event_study",
        metadata=metadata,
        run_id="run_001",
    )

    assert run_dir == tmp_path / "experiments" / "event_study" / "run_001"
    assert run_dir.is_dir()
    assert (run_dir / "research_data.csv").is_file()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "manifest.json").is_file()

    manifest = _read_manifest(run_dir)
    assert manifest["experiment_name"] == "event_study"
    assert manifest["run_id"] == "run_001"
    assert manifest["created_at_utc"].endswith("Z")
    assert manifest["metadata"] == metadata

    artifacts = {artifact["name"]: artifact for artifact in manifest["artifacts"]}
    assert set(artifacts) == set(results)

    for artifact_name, expected_frame in results.items():
        artifact = artifacts[artifact_name]
        assert artifact["path"] == f"{artifact_name}.csv"
        assert not Path(artifact["path"]).is_absolute()
        assert artifact["rows"] == len(expected_frame)
        assert artifact["columns"] == expected_frame.columns.tolist()

        saved_frame = pd.read_csv(run_dir / artifact["path"])
        pd.testing.assert_frame_equal(saved_frame, expected_frame)


def test_save_experiment_results_generates_safe_utc_run_id(tmp_path: Path) -> None:
    run_dir = save_experiment_results(
        results={"artifact": pd.DataFrame({"value": [1]})},
        output_dir=tmp_path,
        experiment_name="generated_run",
    )

    manifest = _read_manifest(run_dir)
    assert run_dir.parent == tmp_path / "generated_run"
    assert run_dir.name == manifest["run_id"]
    assert run_dir.name.endswith("Z")
    assert manifest["created_at_utc"].endswith("Z")


def test_save_experiment_results_does_not_mutate_inputs(tmp_path: Path) -> None:
    results = _sample_results()
    original_keys = list(results.keys())
    original_frames = {
        artifact_name: result.copy(deep=True)
        for artifact_name, result in results.items()
    }
    original_objects = dict(results)

    save_experiment_results(
        results=results,
        output_dir=tmp_path,
        experiment_name="immutability",
        run_id="run_001",
    )

    assert list(results.keys()) == original_keys
    for artifact_name, original_frame in original_frames.items():
        assert results[artifact_name] is original_objects[artifact_name]
        pd.testing.assert_frame_equal(results[artifact_name], original_frame)


def test_empty_results_raise_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        save_experiment_results(
            results={},
            output_dir=tmp_path,
            experiment_name="empty",
            run_id="run_001",
        )


def test_non_dataframe_result_values_raise_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="every result value"):
        save_experiment_results(
            results={"artifact": [1, 2, 3]},
            output_dir=tmp_path,
            experiment_name="invalid_value",
            run_id="run_001",
        )


@pytest.mark.parametrize(
    "experiment_name",
    ["../bad", "bad/name", r"bad\name", "bad name", "bad$name", "bad.name", ""],
)
def test_unsafe_experiment_name_raises_value_error(
    tmp_path: Path,
    experiment_name: str,
) -> None:
    with pytest.raises(ValueError, match="experiment_name"):
        save_experiment_results(
            results={"artifact": pd.DataFrame({"value": [1]})},
            output_dir=tmp_path,
            experiment_name=experiment_name,
            run_id="run_001",
        )


@pytest.mark.parametrize(
    "run_id",
    ["../bad", "bad/name", r"bad\name", "bad name", "bad;name", "bad.name", ""],
)
def test_unsafe_run_id_raises_value_error(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        save_experiment_results(
            results={"artifact": pd.DataFrame({"value": [1]})},
            output_dir=tmp_path,
            experiment_name="safe_experiment",
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "artifact_name",
    ["../bad", "bad/name", r"bad\name", "bad name", "bad|name", "bad.name", ""],
)
def test_unsafe_artifact_names_raise_value_error(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    with pytest.raises(ValueError, match="artifact name"):
        save_experiment_results(
            results={artifact_name: pd.DataFrame({"value": [1]})},
            output_dir=tmp_path,
            experiment_name="safe_experiment",
            run_id="run_001",
        )


def test_existing_run_directory_with_overwrite_false_raises_value_error(
    tmp_path: Path,
) -> None:
    save_experiment_results(
        results={"artifact": pd.DataFrame({"value": [1]})},
        output_dir=tmp_path,
        experiment_name="existing",
        run_id="run_001",
    )

    with pytest.raises(ValueError, match="run directory already exists"):
        save_experiment_results(
            results={"artifact": pd.DataFrame({"value": [2]})},
            output_dir=tmp_path,
            experiment_name="existing",
            run_id="run_001",
        )


def test_existing_run_directory_with_overwrite_true_succeeds(tmp_path: Path) -> None:
    save_experiment_results(
        results={"artifact": pd.DataFrame({"value": [1]})},
        output_dir=tmp_path,
        experiment_name="existing",
        run_id="run_001",
    )

    replacement = pd.DataFrame({"value": [2, 3]})
    run_dir = save_experiment_results(
        results={"artifact": replacement},
        output_dir=tmp_path,
        experiment_name="existing",
        run_id="run_001",
        metadata={"replacement": True},
        overwrite=True,
    )

    saved_frame = pd.read_csv(run_dir / "artifact.csv")
    pd.testing.assert_frame_equal(saved_frame, replacement)

    manifest = _read_manifest(run_dir)
    assert manifest["metadata"] == {"replacement": True}
    assert manifest["artifacts"][0]["rows"] == 2
    assert manifest["artifacts"][0]["columns"] == ["value"]


def test_registry_does_not_create_trading_or_brokerage_columns(
    tmp_path: Path,
) -> None:
    results = {
        "research_output": pd.DataFrame(
            {
                "ticker": ["AAA"],
                "score": [0.5],
            }
        )
    }

    run_dir = save_experiment_results(
        results=results,
        output_dir=tmp_path,
        experiment_name="storage_only",
        run_id="run_001",
    )

    saved_frame = pd.read_csv(run_dir / "research_output.csv")
    manifest = _read_manifest(run_dir)
    saved_columns = [column.lower() for column in saved_frame.columns]
    manifest_columns = [
        column.lower() for column in manifest["artifacts"][0]["columns"]
    ]

    assert saved_frame.columns.tolist() == ["ticker", "score"]
    assert not any(column in FORBIDDEN_CREATED_COLUMNS for column in saved_columns)
    assert not any(column in FORBIDDEN_CREATED_COLUMNS for column in manifest_columns)


def test_registry_does_not_require_network_access_or_external_services(
    tmp_path: Path,
) -> None:
    run_dir = save_experiment_results(
        results={"artifact": pd.DataFrame({"value": [1]})},
        output_dir=tmp_path / "local_only",
        experiment_name="offline",
        run_id="run_001",
    )

    assert run_dir == tmp_path / "local_only" / "offline" / "run_001"
    assert (run_dir / "artifact.csv").is_file()
    assert (run_dir / "manifest.json").is_file()
