"""Accumulate one confusion matrix and calculate segmentation metrics."""

import math

import torch


CITYSCAPES_CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def create_confusion_matrix(
    num_classes: int = 19,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an empty int64 confusion matrix on the requested device."""
    if num_classes <= 0:
        raise ValueError("num_classes должен быть положительным")
    return torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Add a batch to the global matrix; rows are targets, columns predictions."""
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix должна быть квадратной матрицей")
    num_classes = confusion_matrix.shape[0]
    if predictions.ndim == targets.ndim + 1:
        predictions = predictions.argmax(dim=1)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Размер predictions {tuple(predictions.shape)} не совпадает с "
            f"targets {tuple(targets.shape)}"
        )

    device = confusion_matrix.device
    predictions = predictions.detach().reshape(-1).to(
        device=device, dtype=torch.int64
    )
    targets = targets.detach().reshape(-1).to(device=device, dtype=torch.int64)
    not_ignored = targets != ignore_index
    if validate_indices:
        invalid_targets = not_ignored & ((targets < 0) | (targets >= num_classes))
        if torch.any(invalid_targets):
            invalid_value = targets[invalid_targets][0].item()
            raise ValueError(
                f"Маска содержит недопустимый индекс класса: {invalid_value}"
            )
    valid = not_ignored

    valid_predictions = predictions[valid]
    valid_targets = targets[valid]
    if validate_indices:
        invalid_predictions = (valid_predictions < 0) | (
            valid_predictions >= num_classes
        )
        if torch.any(invalid_predictions):
            invalid_value = valid_predictions[invalid_predictions][0].item()
            raise ValueError(f"Предсказан недопустимый индекс класса: {invalid_value}")

    encoded = valid_targets * num_classes + valid_predictions
    batch_matrix = torch.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    confusion_matrix += batch_matrix
    return confusion_matrix


def _mean_without_nan(values: torch.Tensor) -> float:
    valid = ~torch.isnan(values)
    return float(values[valid].mean().item()) if torch.any(valid) else math.nan


def calculate_metrics(confusion_matrix: torch.Tensor) -> dict[str, float | list[float]]:
    """Calculate dataset-level IoU, Dice and accuracy from the final matrix."""
    matrix = confusion_matrix.to(torch.float64)
    true_positive = torch.diag(matrix)
    target_pixels = matrix.sum(dim=1)
    predicted_pixels = matrix.sum(dim=0)

    iou_denominator = target_pixels + predicted_pixels - true_positive
    iou = torch.full_like(true_positive, torch.nan)
    present_for_iou = iou_denominator > 0
    iou[present_for_iou] = true_positive[present_for_iou] / iou_denominator[present_for_iou]

    dice_denominator = target_pixels + predicted_pixels
    dice = torch.full_like(true_positive, torch.nan)
    present_for_dice = dice_denominator > 0
    dice[present_for_dice] = (
        2.0 * true_positive[present_for_dice] / dice_denominator[present_for_dice]
    )

    valid_pixel_count = matrix.sum()
    pixel_accuracy = (
        float(true_positive.sum().item() / valid_pixel_count.item())
        if valid_pixel_count > 0
        else math.nan
    )
    return {
        "miou": _mean_without_nan(iou),
        "macro_dice": _mean_without_nan(dice),
        "pixel_accuracy": pixel_accuracy,
        "iou_per_class": [float(value) for value in iou.tolist()],
    }


def flatten_metrics(
    metrics: dict[str, float | list[float]],
    prefix: str,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Convert nested metric output to flat training-history columns."""
    names = class_names or CITYSCAPES_CLASS_NAMES
    iou_values = metrics["iou_per_class"]
    if not isinstance(iou_values, list) or len(iou_values) != len(names):
        raise ValueError("Число IoU не совпадает с числом имён классов")
    result = {
        f"{prefix}_miou": float(metrics["miou"]),
        f"{prefix}_macro_dice": float(metrics["macro_dice"]),
        f"{prefix}_pixel_accuracy": float(metrics["pixel_accuracy"]),
    }
    for class_name, value in zip(names, iou_values):
        result[f"{prefix}_iou_{class_name}"] = float(value)
    return result
