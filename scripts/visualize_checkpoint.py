"""Visualize one run on a selected official validation image."""

import argparse
import sys
from pathlib import Path

import mlflow
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import (  # noqa: E402
    SUPPORTED_CORRUPTIONS,
    corruption_level,
    corruption_transform,
)
from src.dataset import cityscapes_manifest_dataset  # noqa: E402
from src.experiment import load_run  # noqa: E402
from src.models import create_model  # noqa: E402
from src.tracking import configure_mlflow, read_run_id  # noqa: E402
from src.utils import resolve_path, select_device  # noqa: E402
from src.visualization import save_segmentation_preview  # noqa: E402


def visualize_checkpoint(
    config_path: str | Path,
    index: int,
    condition: str = "clean",
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
    suffix = "clean"
    if condition != "clean":
        level = corruption_level(config, condition, int(severity))
        image_corruption = corruption_transform(condition, level)
        suffix = f"{condition}_s{severity}"

    data = config["data"]
    dataset = cityscapes_manifest_dataset(
        dataset_root=resolve_path(data["root"], project_root),
        images_dir=data["official_val_images"],
        masks_dir=data["official_val_masks"],
        manifest_path=paths.metrics / "official_val_manifest.csv",
        split="val",
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        image_corruption=image_corruption,
        expected_count=500,
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Индекс должен быть от 0 до {len(dataset) - 1}")
    if not paths.best_checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint не найден: {paths.best_checkpoint}")

    device = select_device(str(config["training"].get("device", "auto")))
    checkpoint = torch.load(
        paths.best_checkpoint,
        map_location=device,
        weights_only=False,
    )
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

    sample = dataset[index]
    image = sample["image"]
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(config["training"].get("mixed_precision", True))
        and device.type == "cuda",
    ):
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()

    destination = paths.figures / f"segmentation_{suffix}_index_{index}.png"
    result = save_segmentation_preview(
        image=image,
        ground_truth=sample["mask"],
        prediction=prediction,
        image_id=str(sample["image_id"]),
        output_path=destination,
    )
    configure_mlflow(str(config["tracking"]["experiment_name"]))
    with mlflow.start_run(run_id=read_run_id(paths.run_id)):
        mlflow.log_artifact(str(result), artifact_path="previews")
    print(f"Preview сохранён: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--condition",
        choices=("clean", *SUPPORTED_CORRUPTIONS),
        default="clean",
    )
    parser.add_argument("--severity", type=int, choices=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_checkpoint(args.config, args.index, args.condition, args.severity)


if __name__ == "__main__":
    main()
