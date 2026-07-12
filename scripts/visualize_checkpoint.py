"""Show selected official-validation predictions without saving image files."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import SUPPORTED_CORRUPTIONS  # noqa: E402
from src.inference import (  # noqa: E402
    build_official_val_dataset,
    load_inference_run,
    predict_masks,
)
from src.visualization import create_segmentation_preview  # noqa: E402


def visualize_checkpoints(
    config_path: str | Path,
    indices: list[int],
    condition: str = "clean",
    severity: int | None = None,
) -> list[plt.Figure]:
    """Build four-panel figures in memory for the selected indices."""
    if not indices:
        raise ValueError("indices должен содержать хотя бы один индекс")
    selected_indices = [int(index) for index in indices]
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("indices не должен содержать повторяющиеся значения")

    run = load_inference_run(config_path)
    dataset, _ = build_official_val_dataset(run, condition, severity)
    for index in selected_indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Индекс должен быть от 0 до {len(dataset) - 1}: {index}")

    figures: list[plt.Figure] = []
    batch_size = int(run.config["evaluation"].get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("evaluation.batch_size должен быть > 0")
    for start in range(0, len(selected_indices), batch_size):
        batch_indices = selected_indices[start : start + batch_size]
        samples = [dataset[index] for index in batch_indices]
        images = torch.stack([sample["image"] for sample in samples])
        predictions = predict_masks(run, images)
        for sample, prediction in zip(samples, predictions):
            figures.append(
                create_segmentation_preview(
                    image=sample["image"],
                    ground_truth=sample["mask"],
                    prediction=prediction,
                    image_id=str(sample["image_id"]),
                )
            )
    return figures


def visualize_checkpoint(
    config_path: str | Path,
    index: int,
    condition: str = "clean",
    severity: int | None = None,
) -> plt.Figure:
    """Build one four-panel figure in memory."""
    return visualize_checkpoints(config_path, [index], condition, severity)[0]


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
    figure = visualize_checkpoint(
        args.config,
        args.index,
        args.condition,
        args.severity,
    )
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
