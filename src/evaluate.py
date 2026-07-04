"""Reusable evaluation loop for one segmentation checkpoint."""

import math
import time

import torch
from torch.utils.data import DataLoader

from src.metrics import calculate_metrics, create_confusion_matrix, update_confusion_matrix


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    use_amp: bool,
    label: str,
) -> tuple[dict, torch.Tensor, dict[str, float]]:
    model.eval()
    confusion = create_confusion_matrix(num_classes, device=device)
    total_seconds = 0.0
    image_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_seconds += time.perf_counter() - started_at
        image_count += int(images.shape[0])
        update_confusion_matrix(
            confusion,
            logits,
            targets,
            ignore_index=ignore_index,
            validate_indices=False,
        )
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(dataloader):
            print(f"[{label}] batch {batch_index}/{len(dataloader)}", flush=True)

    if image_count == 0 or confusion.sum().item() == 0:
        raise ValueError("Оценка не обработала ни одного валидного пикселя")
    metrics = calculate_metrics(confusion)
    if not math.isfinite(float(metrics["miou"])):
        raise ValueError("mIoU не является конечным числом")
    resources = {
        "num_images": float(image_count),
        "mean_inference_ms_per_image": total_seconds * 1000.0 / image_count,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    return metrics, confusion.detach().cpu(), resources
