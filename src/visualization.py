"""Create clear Cityscapes prediction previews from tensors."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.dataset import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD


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


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized CHW tensor to an RGB image in 0..1."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Ожидалось изображение CHW с 3 каналами, получено {tuple(image.shape)}"
        )
    array = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.clip(array * std + mean, 0.0, 1.0)


def colorize_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map Cityscapes trainId 0..18 to colors; show ignore_index as black."""
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Ожидалась двумерная маска, получено {array.shape}")
    invalid = np.setdiff1d(
        np.unique(array), np.array([*range(len(CITYSCAPES_COLORS)), IGNORE_INDEX])
    )
    if invalid.size:
        raise ValueError(
            "Маска содержит недопустимые trainId: "
            + ", ".join(map(str, invalid[:10].tolist()))
        )
    colored = np.zeros((*array.shape, 3), dtype=np.uint8)
    valid = (array >= 0) & (array < len(CITYSCAPES_COLORS))
    colored[valid] = CITYSCAPES_COLORS[array[valid].astype(np.int64)]
    return colored


def save_segmentation_preview(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    image_id: str,
    output_path: str | Path,
) -> Path:
    """Save input, ground truth, prediction and prediction overlay in one PNG."""
    rgb = denormalize_image(image)
    target = ground_truth.detach().cpu().numpy()
    predicted = prediction.detach().cpu().numpy()
    if target.shape != predicted.shape or target.shape != rgb.shape[:2]:
        raise ValueError(
            "Размеры изображения, ground truth и prediction не совпадают: "
            f"{rgb.shape[:2]}, {target.shape}, {predicted.shape}"
        )
    ground_truth_color = colorize_mask(target)
    prediction_color = colorize_mask(predicted)
    overlay = 0.58 * rgb + 0.42 * (prediction_color.astype(np.float32) / 255.0)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [rgb, ground_truth_color, prediction_color, overlay]
    titles = ["Исходное изображение", "Ground truth", "Prediction", "Наложение"]
    for axis, panel, title in zip(axes, panels, titles):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(str(image_id), fontsize=10)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Не удалось сохранить preview: {destination}")
    return destination
