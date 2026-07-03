"""Backfill a completed baseline run from CSV and checkpoints into MLflow."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking import (  # noqa: E402
    check_mlflow_connection,
    log_artifact_safe,
    log_metrics_safe,
    mlflow_run,
)
from src.utils import load_yaml, resolve_path  # noqa: E402


MODELS = ["unet", "deeplabv3plus", "pspnet"]


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_backfill_run(
    experiment_id: str,
    model_name: str,
    seed: int,
) -> Any | None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    filter_string = (
        f"tags.model_name = '{model_name}' AND "
        f"tags.seed = '{seed}' AND "
        "tags.source = 'backfill'"
    )
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def backfill(config_path: str | Path, model_name: str) -> str:
    import mlflow

    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    project_root = config_file.parent.parent
    training = config["training"]
    seed = int(config["seed"])
    epochs = int(training["epochs"])
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    history_dir = resolve_path(
        training.get("history_dir", "outputs/metrics"), project_root
    )

    history_path = history_dir / f"training_history_{model_name}.csv"
    environment_path = history_dir / "training_environment.json"
    best_path = checkpoint_dir / f"{model_name}_best.pt"
    last_path = checkpoint_dir / f"{model_name}_last.pt"
    run_id_path = history_dir / f"mlflow_run_id_{model_name}.txt"
    for required in (history_path, environment_path, best_path, last_path):
        if not required.is_file():
            raise FileNotFoundError(f"Обязательный artifact не найден: {required}")

    history = pd.read_csv(history_path)
    if history.empty or "epoch" not in history.columns:
        raise ValueError(f"CSV не содержит эпох: {history_path}")
    if int(history["epoch"].max()) < epochs:
        raise ValueError(
            f"Обучение не завершено: CSV содержит эпохи только до "
            f"{int(history['epoch'].max())}, ожидалось {epochs}"
        )
    best_checkpoint = load_checkpoint(best_path)
    last_checkpoint = load_checkpoint(last_path)
    if int(last_checkpoint.get("epoch", -1)) < epochs:
        raise ValueError(f"Last checkpoint не достиг эпохи {epochs}: {last_path}")
    if last_checkpoint.get("model_name") != model_name:
        raise ValueError("Имя модели в last checkpoint не совпадает с CLI")
    if best_checkpoint.get("model_name") != model_name:
        raise ValueError("Имя модели в best checkpoint не совпадает с CLI")

    experiment_name = config.get("tracking", {}).get(
        "experiment_name", "cityscapes_robustness"
    )
    connection = check_mlflow_connection(experiment_name)
    existing_run = find_backfill_run(
        connection["experiment_id"], model_name, seed
    )
    if existing_run is not None:
        existing_run_id = existing_run.info.run_id
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(existing_run_id + "\n", encoding="utf-8")
        if existing_run.data.tags.get("status") == "completed":
            print(
                f"Backfill уже выполнен, повторный run не создан: {existing_run_id}"
            )
            return existing_run_id

    checkpoint_config = last_checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("В last checkpoint отсутствует словарь config")
    run_name = f"baseline_{model_name}_seed{seed}"
    tags = {
        "model_name": model_name,
        "seed": str(seed),
        "source": "backfill",
        "status": "running",
    }
    with mlflow_run(
        experiment_name,
        run_name,
        checkpoint_config,
        run_id_path=run_id_path,
        existing_run_id=(existing_run.info.run_id if existing_run is not None else None),
        tags=tags,
    ) as mlflow_module:
        if mlflow_module is None:
            raise RuntimeError("MLflow недоступен: backfill не выполнен")
        for _, row in history.iterrows():
            step = int(row["epoch"])
            metrics = {
                str(key): float(value)
                for key, value in row.items()
                if key != "epoch"
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            }
            log_metrics_safe(mlflow_module, metrics, step)
        log_artifact_safe(mlflow_module, history_path, "training")
        log_artifact_safe(mlflow_module, environment_path, "environment")
        log_artifact_safe(mlflow_module, best_path, "checkpoints")
        log_artifact_safe(mlflow_module, last_path, "checkpoints")
        mlflow.set_tags(
            {
                "model_name": model_name,
                "seed": str(seed),
                "source": "backfill",
                "status": "completed",
            }
        )
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("MLflow run неожиданно завершился во время backfill")
        run_id = active_run.info.run_id
    print(f"Backfill завершён: run_id={run_id}")
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        backfill(args.config, args.model)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка backfill: {error}") from error


if __name__ == "__main__":
    main()
