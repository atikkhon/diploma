"""Shared checkpoint loading and official-validation inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.corruptions import (
    SUPPORTED_CORRUPTIONS,
    corruption_level,
    corruption_transform,
)
from src.dataset import CityscapesDataset, cityscapes_manifest_dataset
from src.experiment import RunPaths, load_run
from src.models import create_model
from src.utils import resolve_path, select_device


@dataclass
class InferenceRun:
    config: dict[str, Any]
    project_root: Path
    paths: RunPaths
    device: torch.device
    model: nn.Module
    checkpoint_epoch: int


def load_inference_run(config_path: str | Path) -> InferenceRun:
    """Load one run's best checkpoint exactly as evaluation does."""
    config, project_root, paths = load_run(config_path)
    if not paths.best_checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint не найден: {paths.best_checkpoint}")

    data = config["data"]
    device = select_device(str(config["training"].get("device", "auto")))
    checkpoint = torch.load(
        paths.best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint не содержит model_state_dict: {paths.best_checkpoint}"
        )
    if "epoch" not in checkpoint:
        raise ValueError(f"Checkpoint не содержит epoch: {paths.best_checkpoint}")

    model_settings = dict(config["model"])
    model_name = str(model_settings["name"]).lower()
    model_settings["encoder_weights"] = None
    model = create_model(
        model_name,
        classes=int(data["num_classes"]),
        settings=model_settings,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return InferenceRun(
        config=config,
        project_root=project_root,
        paths=paths,
        device=device,
        model=model,
        checkpoint_epoch=int(checkpoint["epoch"]),
    )


def build_official_val_dataset(
    run: InferenceRun,
    condition: str,
    severity: int | None,
) -> tuple[CityscapesDataset, dict[str, Any]]:
    """Build the same official-validation input used by evaluation."""
    allowed_conditions = {"clean", *SUPPORTED_CORRUPTIONS}
    if condition not in allowed_conditions:
        raise ValueError(
            f"condition должен быть одним из: {', '.join(sorted(allowed_conditions))}"
        )
    if condition == "clean" and severity is not None:
        raise ValueError("Clean не использует severity")
    if condition != "clean" and severity not in {1, 2, 3}:
        raise ValueError(f"Для {condition} выберите severity 1, 2 или 3")

    level: dict[str, Any] = {}
    image_corruption = None
    if condition != "clean":
        level = corruption_level(run.config, condition, int(severity))
        image_corruption = corruption_transform(condition, level)

    data = run.config["data"]
    dataset = cityscapes_manifest_dataset(
        dataset_root=resolve_path(data["root"], run.project_root),
        images_dir=data["official_val_images"],
        masks_dir=data["official_val_masks"],
        manifest_path=run.paths.metrics / "official_val_manifest.csv",
        split="val",
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        image_corruption=image_corruption,
        expected_count=500,
    )
    return dataset, level


def predict_masks(run: InferenceRun, images: torch.Tensor) -> torch.Tensor:
    """Return trainId predictions for a BCHW tensor."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Ожидался tensor BCHW, получено {tuple(images.shape)}")
    with torch.inference_mode(), torch.autocast(
        device_type=run.device.type,
        dtype=torch.float16,
        enabled=bool(run.config["training"].get("mixed_precision", True))
        and run.device.type == "cuda",
    ):
        predictions = run.model(images.to(run.device)).argmax(dim=1)
    return predictions.cpu()
