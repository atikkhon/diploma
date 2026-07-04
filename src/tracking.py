"""Required SQLite-backed MLflow tracking."""

import math
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def flatten_parameters(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_parameters(value, full_key))
        elif isinstance(value, (list, tuple)):
            result[full_key] = ",".join(map(str, value))
        elif value is None:
            result[full_key] = "null"
        else:
            result[full_key] = value
    return result


def tracking_settings() -> tuple[str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "").strip().rstrip("/")
    if not tracking_uri.startswith("sqlite:"):
        raise ValueError("MLFLOW_TRACKING_URI должен быть SQLite URI")
    if not artifact_root.startswith("file:"):
        raise ValueError("MLFLOW_ARTIFACT_ROOT должен быть file:// URI")
    return tracking_uri, artifact_root


def configure_mlflow(experiment_name: str) -> Any:
    tracking_uri, artifact_root = tracking_settings()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root,
        )
        experiment = client.get_experiment(experiment_id)
    actual_location = str(experiment.artifact_location).rstrip("/")
    if actual_location != artifact_root:
        raise ValueError(
            "MLflow experiment использует другой artifact location: "
            f"{actual_location}; ожидалось {artifact_root}"
        )
    return experiment


def check_mlflow_connection(experiment_name: str) -> dict[str, str]:
    experiment = configure_mlflow(experiment_name)
    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_id": str(experiment.experiment_id),
        "experiment_name": experiment.name,
        "artifact_location": str(experiment.artifact_location),
    }


def read_run_id(path: str | Path) -> str:
    run_id_file = Path(path)
    if not run_id_file.is_file():
        raise FileNotFoundError(f"MLflow run ID не найден: {run_id_file}")
    run_id = run_id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"MLflow run ID пуст: {run_id_file}")
    return run_id


def write_run_id(path: str | Path, run_id: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(run_id + "\n", encoding="utf-8")


def finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def log_artifacts(paths: list[str | Path], artifact_path: str | None = None) -> None:
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"MLflow artifact не найден: {path}")
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
