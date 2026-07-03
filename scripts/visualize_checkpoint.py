"""Visualize one best checkpoint on a deterministic internal-dev image."""

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CityscapesDataset  # noqa: E402
from src.models import create_model  # noqa: E402
from src.utils import load_yaml, resolve_path, select_device  # noqa: E402
from src.visualization import save_segmentation_preview  # noqa: E402


MODEL_NAMES = ("unet", "deeplabv3plus", "pspnet")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a training checkpoint with clear validation errors."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint не найден: {path}. Сначала обучите эту модель."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Файл не является checkpoint проекта: {path}")
    return checkpoint


def visualize_checkpoint(
    config_path: str | Path,
    model_name: str,
    index: int = 0,
    output_path: str | Path | None = None,
) -> Path:
    """Run one prediction on internal dev and save a four-panel PNG."""
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    project_root = config_file.parent.parent
    data = config.get("data", {})
    models = config.get("models", {})
    training = config.get("training", {})
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_NAMES:
        raise ValueError(f"model должен быть одним из: {', '.join(MODEL_NAMES)}")

    dataset = CityscapesDataset(
        manifest_path=resolve_path(data["split_file"], project_root),
        dataset_root=resolve_path(data["root"], project_root),
        split="dev",
        train=False,
        width=int(data.get("image_width", 384)),
        height=int(data.get("image_height", 192)),
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"Индекс dev-примера {index} вне диапазона 0..{len(dataset) - 1}"
        )
    row = dataset.rows.iloc[index]
    for column in ("image_path", "mask_path"):
        parts = {part.lower() for part in Path(str(row[column])).parts}
        if "val" in parts or "train" not in parts:
            raise ValueError(
                f"Preview разрешён только на internal dev из official train; "
                f"получен путь {row[column]}"
            )

    device = select_device(str(training.get("device", "auto")))
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    checkpoint_path = checkpoint_dir / f"{normalized_name}_best.pt"
    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_name = str(checkpoint.get("model_name", normalized_name)).lower()
    if saved_name != normalized_name:
        raise ValueError(
            f"В {checkpoint_path} сохранена модель {saved_name}, "
            f"но запрошена {normalized_name}"
        )

    print(f"Загрузка {normalized_name} из {checkpoint_path} на {device}...", flush=True)
    model = create_model(
        normalized_name,
        classes=int(data.get("num_classes", 19)),
        encoder_name=str(models.get("encoder", "resnet34")),
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    sample = dataset[index]
    image = sample["image"]
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
    ):
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1)[0].cpu()

    destination = (
        resolve_path(output_path, project_root)
        if output_path is not None
        else project_root
        / "outputs"
        / "figures"
        / f"segmentation_preview_{normalized_name}.png"
    )
    result = save_segmentation_preview(
        image=image,
        ground_truth=sample["mask"],
        prediction=prediction,
        image_id=str(sample["image_id"]),
        output_path=destination,
    )
    print(f"Preview сохранён: {result}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        visualize_checkpoint(args.config, args.model, args.index, args.output)
    except (FileNotFoundError, KeyError, ValueError, IndexError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка визуализации checkpoint: {error}") from error


if __name__ == "__main__":
    main()
