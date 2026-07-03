"""Use optional SQLite-backed MLflow without making training depend on it."""

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


def _tracking_settings() -> tuple[str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "").strip()
    if not tracking_uri:
        raise ValueError("Переменная MLFLOW_TRACKING_URI не задана")
    if not tracking_uri.startswith("sqlite:"):
        raise ValueError(
            "MLFLOW_TRACKING_URI должен использовать SQLite, например "
            "sqlite:////content/drive/MyDrive/.../mlflow.db"
        )
    if not artifact_root:
        raise ValueError("Переменная MLFLOW_ARTIFACT_ROOT не задана")
    if not artifact_root.startswith("file:"):
        raise ValueError("MLFLOW_ARTIFACT_ROOT должен быть file:// URI")
    return tracking_uri, artifact_root.rstrip("/")


def get_or_create_experiment(
    mlflow_module: Any,
    experiment_name: str,
    artifact_root: str,
) -> Any:
    """Create an experiment with an explicit artifact root or verify the existing one."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_module.get_tracking_uri())
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root,
        )
        experiment = client.get_experiment(experiment_id)
    actual_location = str(experiment.artifact_location).rstrip("/")
    if actual_location != artifact_root.rstrip("/"):
        warnings.warn(
            "MLflow experiment уже существует с другим artifact location: "
            f"{actual_location}; ожидалось {artifact_root}. "
            "MLflow не позволяет изменить его после создания.",
            RuntimeWarning,
        )
    return experiment


def check_mlflow_connection(experiment_name: str) -> dict[str, str]:
    """Initialize SQLite and return connection details for a Colab smoke test."""
    import mlflow

    tracking_uri, artifact_root = _tracking_settings()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = get_or_create_experiment(mlflow, experiment_name, artifact_root)
    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_id": str(experiment.experiment_id),
        "experiment_name": experiment.name,
        "artifact_location": str(experiment.artifact_location),
    }


def _read_run_id(run_id_path: Path | None) -> str | None:
    if run_id_path is None or not run_id_path.is_file():
        return None
    value = run_id_path.read_text(encoding="utf-8").strip()
    return value or None


def _save_run_id(run_id_path: Path | None, run_id: str) -> None:
    if run_id_path is None:
        return
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.write_text(run_id + "\n", encoding="utf-8")


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    parameters: dict[str, Any],
    run_id_path: str | Path | None = None,
    resume_existing: bool = False,
    tags: dict[str, str] | None = None,
) -> Iterator[Any | None]:
    """Start or resume an optional SQLite MLflow run and persist its run ID."""
    mlflow_api = None
    active_mlflow = None
    run_started = False
    run_id_file = Path(run_id_path) if run_id_path is not None else None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri, artifact_root = _tracking_settings()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow_api = mlflow
        experiment = get_or_create_experiment(
            mlflow, experiment_name, artifact_root
        )

        saved_run_id = _read_run_id(run_id_file) if resume_existing else None
        active_run = None
        if resume_existing and saved_run_id is None:
            warnings.warn(
                f"MLflow run_id для resume не найден: {run_id_file}. "
                "Будет создан новый MLflow run; обучение продолжится.",
                RuntimeWarning,
            )
        if saved_run_id is not None:
            try:
                client = MlflowClient(tracking_uri=tracking_uri)
                saved_run = client.get_run(saved_run_id)
                if str(saved_run.info.experiment_id) != str(experiment.experiment_id):
                    raise ValueError("run относится к другому experiment")
                active_run = mlflow.start_run(run_id=saved_run_id, tags=tags)
            except Exception as error:
                warnings.warn(
                    f"Не удалось продолжить MLflow run {saved_run_id}: {error}. "
                    "Будет создан новый run; CSV и checkpoint не затронуты.",
                    RuntimeWarning,
                )

        if active_run is None:
            active_run = mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name,
                tags=tags,
            )
        run_started = True
        _save_run_id(run_id_file, active_run.info.run_id)
        try:
            mlflow.log_params(flatten_parameters(parameters))
        except Exception as error:
            warnings.warn(f"Не удалось записать параметры в MLflow: {error}")
        active_mlflow = mlflow
        print(f"MLflow run_id: {active_run.info.run_id}", flush=True)
    except Exception as error:  # MLflow must never stop the experiment.
        warnings.warn(
            f"MLflow недоступен, обучение продолжится с CSV: {error}",
            RuntimeWarning,
        )

    try:
        yield active_mlflow
    except BaseException:
        if run_started and mlflow_api is not None:
            try:
                mlflow_api.end_run(status="FAILED")
            except Exception as error:
                warnings.warn(f"Не удалось отметить MLflow run как FAILED: {error}")
        raise
    else:
        if run_started and mlflow_api is not None:
            try:
                mlflow_api.end_run(status="FINISHED")
            except Exception as error:
                warnings.warn(f"Не удалось завершить MLflow run: {error}")


def log_metrics_safe(
    mlflow_module: Any | None,
    metrics: dict[str, float],
    step: int,
) -> None:
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


def log_artifact_safe(
    mlflow_module: Any | None,
    path: str | Path,
    artifact_path: str | None = None,
) -> None:
    """Log one existing file and turn MLflow errors into warnings."""
    if mlflow_module is None:
        return
    artifact = Path(path)
    if not artifact.is_file():
        warnings.warn(f"MLflow artifact не найден: {artifact}")
        return
    try:
        mlflow_module.log_artifact(str(artifact), artifact_path=artifact_path)
    except Exception as error:
        warnings.warn(f"Не удалось записать artifact в MLflow: {error}")
