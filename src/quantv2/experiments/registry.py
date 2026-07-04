from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_safe_name(name: str, field_name: str) -> None:
    if not isinstance(name, str) or not SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, underscores, or hyphens"
        )


def save_experiment_results(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/experiments",
    experiment_name: str = "experiment",
    metadata: dict | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Save experiment result DataFrames and a manifest to a local run directory.

    This registry function only persists outputs that were already produced by an
    experiment runner. It does not fetch data, create labels or features, train
    models, create trading signals, or calculate profitability.
    """
    if not results:
        raise ValueError("results must not be empty")

    _validate_safe_name(experiment_name, "experiment_name")
    for artifact_name, result in results.items():
        _validate_safe_name(artifact_name, "artifact name")
        if not isinstance(result, pd.DataFrame):
            raise ValueError("every result value must be a pandas DataFrame")

    created_at = datetime.now(timezone.utc)
    created_at_utc = created_at.isoformat().replace("+00:00", "Z")
    if run_id is None:
        run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    else:
        _validate_safe_name(run_id, "run_id")

    run_dir = Path(output_dir) / experiment_name / run_id
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"run path already exists and is not a directory: {run_dir}")
        if not overwrite:
            raise ValueError(f"run directory already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True)

    artifacts = []
    for artifact_name, result in results.items():
        artifact_path = run_dir / f"{artifact_name}.csv"
        result.to_csv(artifact_path, index=False)
        artifacts.append(
            {
                "name": artifact_name,
                "path": artifact_path.relative_to(run_dir).as_posix(),
                "rows": int(len(result)),
                "columns": list(result.columns),
            }
        )

    manifest = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "metadata": {} if metadata is None else dict(metadata),
        "artifacts": artifacts,
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return run_dir
