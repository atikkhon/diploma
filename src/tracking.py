"""Log to MLflow when available without making training depend on MLflow."""

import math
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def flatten_parameters(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration dictionaries for MLflow parameters."""
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


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    parameters: dict[str, Any],
) -> Iterator[Any | None]:
    """Start an optional run; URI is read only from MLFLOW_TRACKING_URI."""
    mlflow_api = None
    active_mlflow = None
    run_started = False
    try:
        import mlflow

        mlflow_api = mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        run_started = True
        mlflow.log_params(flatten_parameters(parameters))
        active_mlflow = mlflow
    except Exception as error:  # MLflow must never stop the experiment.
        warnings.warn(
            f"MLflow недоступен, обучение продолжится с CSV: {error}",
            RuntimeWarning,
        )

    try:
        yield active_mlflow
    finally:
        if run_started and mlflow_api is not None:
            try:
                mlflow_api.end_run()
            except Exception as error:
                warnings.warn(f"Не удалось завершить MLflow run: {error}")


def log_metrics_safe(mlflow_module: Any | None, metrics: dict[str, float], step: int) -> None:
    """Log finite numeric metrics and turn MLflow errors into warnings."""
    if mlflow_module is None:
        return
    finite_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    try:
        mlflow_module.log_metrics(finite_metrics, step=step)
    except Exception as error:
        warnings.warn(f"Не удалось записать метрики в MLflow: {error}")


def log_artifact_safe(mlflow_module: Any | None, path: str | Path) -> None:
    """Log one existing file and turn MLflow errors into warnings."""
    if mlflow_module is None:
        return
    artifact = Path(path)
    if not artifact.is_file():
        warnings.warn(f"MLflow artifact не найден: {artifact}")
        return
    try:
        mlflow_module.log_artifact(str(artifact))
    except Exception as error:
        warnings.warn(f"Не удалось записать artifact в MLflow: {error}")
