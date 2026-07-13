"""Check a dataset manifest and visualize eight deterministic samples."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    CityscapesDataset,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_data_config(config_path: str | Path) -> dict:
    path = resolve_project_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("В YAML-конфигурации отсутствует словарь data")
    return config


def denormalize_image(image_tensor) -> np.ndarray:
    """Convert normalized CHW tensor back to displayable RGB float image."""
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.clip(image * std + mean, 0.0, 1.0)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map trainId 0..18 to Cityscapes colors and ignore pixels to black."""
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(CITYSCAPES_COLORS))
    color[valid] = CITYSCAPES_COLORS[mask[valid]]
    color[mask == IGNORE_INDEX] = 0
    return color


def check_dataset(
    config_path: str | Path = "configs/experiment.yaml",
    split: str = "dev",
    output_path: str | Path = "outputs/figures/dataset_check.png",
) -> Path:
    """Create an 8x3 dataset check with image, mask and overlay."""
    config = load_data_config(config_path)
    data = config["data"]
    required = ["root", "split_file", "image_width", "image_height"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"В data отсутствуют параметры: {missing}")

    dataset = CityscapesDataset(
        manifest_path=resolve_project_path(data["split_file"]),
        dataset_root=resolve_project_path(data["root"]),
        split=split,
        train=False,
        width=int(data["image_width"]),
        height=int(data["image_height"]),
    )
    if len(dataset) < 8:
        raise ValueError(
            f"Для проверки датасета нужны минимум 8 примеров в split={split}, "
            f"найдено: {len(dataset)}"
        )

    # Evenly spaced deterministic indices show more than one city when possible.
    indices = np.linspace(0, len(dataset) - 1, num=8, dtype=int)
    figure, axes = plt.subplots(8, 3, figsize=(12, 24), constrained_layout=True)
    column_titles = ["Изображение", "Ground truth", "Наложение"]

    for row_number, dataset_index in enumerate(indices):
        sample = dataset[int(dataset_index)]
        image = denormalize_image(sample["image"])
        mask = sample["mask"].detach().cpu().numpy()
        colored_mask = colorize_mask(mask)
        overlay = image.copy()
        valid = mask != IGNORE_INDEX
        overlay[valid] = (
            0.6 * image[valid] + 0.4 * (colored_mask[valid].astype(np.float32) / 255.0)
        )

        panels = [image, colored_mask, overlay]
        for column_number, panel in enumerate(panels):
            axis = axes[row_number, column_number]
            axis.imshow(panel)
            axis.axis("off")
            if row_number == 0:
                axis.set_title(column_titles[column_number])
        axes[row_number, 0].set_ylabel(
            str(sample["image_id"]), rotation=0, ha="right", va="center", fontsize=8
        )

    destination = resolve_project_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"PNG-файл не был сохранён: {destination}")
    print(f"Проверка датасета сохранена: {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=["train", "dev", "val"], default="dev")
    parser.add_argument(
        "--output", default="outputs/figures/dataset_check.png"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_dataset(args.config, args.split, args.output)


if __name__ == "__main__":
    main()
