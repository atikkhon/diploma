"""Evaluate one run on clean images or one manually selected corruption level."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import (  # noqa: E402
    SUPPORTED_CORRUPTIONS,
    corruption_level,
    corruption_parameters,
    corruption_transform,
)
from src.dataset import cityscapes_manifest_dataset  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.experiment import load_run  # noqa: E402
from src.metrics import CITYSCAPES_CLASS_NAMES  # noqa: E402
from src.models import create_model  # noqa: E402
from src.tracking import configure_mlflow, finite_metrics, read_run_id  # noqa: E402
from src.utils import (  # noqa: E402
    make_dataloader_generator,
    resolve_path,
    seed_everything,
    seed_worker,
    select_device,
)


def append_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    new_rows = pd.DataFrame(rows)
    if destination.is_file():
        previous = pd.read_csv(destination)
        new_rows = pd.concat([previous, new_rows], ignore_index=True)
    new_rows.to_csv(destination, index=False, encoding="utf-8")


def load_best_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint не найден: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "model_name", "epoch"}
    if not isinstance(checkpoint, dict) or not required <= set(checkpoint):
        raise ValueError(f"Некорректный checkpoint: {path}")
    return checkpoint


def evaluate_run(
    config_path: str | Path,
    condition: str,
    severity: int | None = None,
) -> Path:
    config, project_root, paths = load_run(config_path)
    paths.create()
    allowed_conditions = {"clean", *SUPPORTED_CORRUPTIONS}
    if condition not in allowed_conditions:
        raise ValueError(
            f"condition должен быть одним из: {', '.join(sorted(allowed_conditions))}"
        )
    if condition != "clean" and severity not in {1, 2, 3}:
        raise ValueError(f"Для {condition} выберите severity 1, 2 или 3")

    image_corruption = None
    corruption_params: dict[str, float | int] = {}
    if condition != "clean":
        level = corruption_level(config, condition, int(severity))
        corruption_params = corruption_parameters(condition, level)
        image_corruption = corruption_transform(condition, level)

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    training = config["training"]
    evaluation = config["evaluation"]
    data = config["data"]
    model_settings = dict(config["model"])
    model_name = str(model_settings["name"]).lower()
    device = select_device(str(training.get("device", "auto")))
    manifest_path = paths.metrics / "official_val_manifest.csv"
    dataset = cityscapes_manifest_dataset(
        dataset_root=resolve_path(data["root"], project_root),
        images_dir=data["official_val_images"],
        masks_dir=data["official_val_masks"],
        manifest_path=manifest_path,
        split="val",
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        image_corruption=image_corruption,
        expected_count=500,
    )
    num_workers = int(training.get("num_workers", 0))
    dataloader = DataLoader(
        dataset,
        batch_size=int(evaluation["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
        persistent_workers=num_workers > 0,
    )

    checkpoint = load_best_checkpoint(paths.best_checkpoint)
    if str(checkpoint["model_name"]).lower() != model_name:
        raise ValueError("Checkpoint относится к другой модели")
    model_settings["encoder_weights"] = None
    model = create_model(
        model_name,
        classes=int(data["num_classes"]),
        settings=model_settings,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics, confusion, resources = evaluate_model(
        model,
        dataloader,
        device,
        num_classes=int(data["num_classes"]),
        ignore_index=int(data["ignore_index"]),
        use_amp=bool(training.get("mixed_precision", True)),
        label=f"{model_name}/{condition}",
    )

    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    clean_miou = float(metrics["miou"])
    delta_miou = 0.0
    retention = 1.0
    if condition != "clean":
        if not paths.evaluations.is_file():
            raise FileNotFoundError("Сначала выполните clean evaluation этого запуска")
        previous = pd.read_csv(paths.evaluations)
        clean_rows = previous.loc[previous["condition"] == "clean"]
        if clean_rows.empty:
            raise ValueError("Сначала выполните clean evaluation этого запуска")
        clean_miou = float(clean_rows.iloc[-1]["miou"])
        delta_miou = clean_miou - float(metrics["miou"])
        retention = float(metrics["miou"]) / clean_miou

    summary = {
        "evaluation_id": evaluation_id,
        "run_name": str(config["run"]["name"]),
        "model": model_name,
        "condition": condition,
        "severity": 0 if severity is None else severity,
        **corruption_params,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "miou": float(metrics["miou"]),
        "macro_dice": float(metrics["macro_dice"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "clean_miou": clean_miou,
        "delta_miou": delta_miou,
        "retention": retention,
        **resources,
    }
    per_class_rows = [
        {
            "evaluation_id": evaluation_id,
            "run_name": str(config["run"]["name"]),
            "model": model_name,
            "condition": condition,
            "severity": 0 if severity is None else severity,
            "class_id": class_id,
            "class_name": class_name,
            "iou": float(iou),
        }
        for class_id, (class_name, iou) in enumerate(
            zip(CITYSCAPES_CLASS_NAMES, metrics["iou_per_class"])
        )
    ]
    evaluation_dir = paths.metrics / "evaluations" / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    summary_path = evaluation_dir / "summary.csv"
    per_class_path = evaluation_dir / "per_class_iou.csv"
    confusion_path = evaluation_dir / "confusion_matrix.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False, encoding="utf-8")
    pd.DataFrame(
        confusion.numpy(),
        index=CITYSCAPES_CLASS_NAMES,
        columns=CITYSCAPES_CLASS_NAMES,
    ).to_csv(confusion_path, index_label="target_class", encoding="utf-8")
    append_csv([summary], paths.evaluations)
    append_csv(per_class_rows, paths.per_class)

    parent_run_id = read_run_id(paths.run_id)
    experiment = configure_mlflow(str(config["tracking"]["experiment_name"]))
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"{config['run']['name']}_{condition}_{evaluation_id}",
        tags={
            "mlflow.parentRunId": parent_run_id,
            "model": model_name,
            "stage": "evaluation",
            "condition": condition,
        },
    ):
        mlflow.log_params(
            {
                "run_name": str(config["run"]["name"]),
                "condition": condition,
                "severity": 0 if severity is None else severity,
                **corruption_params,
                "checkpoint_epoch": int(checkpoint["epoch"]),
            }
        )
        mlflow.log_metrics(finite_metrics(summary))
        for artifact in (summary_path, per_class_path, confusion_path):
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Evaluation CSV: {paths.evaluations}")
    return paths.evaluations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--condition",
        choices=("clean", *SUPPORTED_CORRUPTIONS),
        required=True,
    )
    parser.add_argument("--severity", type=int, choices=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_run(args.config, args.condition, args.severity)


if __name__ == "__main__":
    main()
