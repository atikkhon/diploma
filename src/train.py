"""Train and validate segmentation models with global dataset metrics."""

import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import (
    calculate_metrics,
    create_confusion_matrix,
    flatten_metrics,
    update_confusion_matrix,
)


def _mixed_precision_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> dict[str, float | list[float]]:
    """Train for one epoch and calculate metrics from one global matrix."""
    model.train()
    confusion = create_confusion_matrix(num_classes)
    weighted_loss_sum = 0.0
    valid_pixel_sum = 0
    amp_enabled = _mixed_precision_enabled(device, use_amp)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid_pixels = int((targets != ignore_index).sum().item())
        if valid_pixels == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Получен нечисловой train loss: {loss.item()}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_loss_sum += float(loss.item()) * valid_pixels
        valid_pixel_sum += valid_pixels
        update_confusion_matrix(confusion, logits, targets, ignore_index)

    if valid_pixel_sum == 0:
        raise ValueError("Train-набор не содержит ни одного неигнорируемого пикселя")
    result = calculate_metrics(confusion)
    result["loss"] = weighted_loss_sum / valid_pixel_sum
    return result


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> dict[str, float | list[float]]:
    """Evaluate the complete set before calculating any segmentation metric."""
    model.eval()
    confusion = create_confusion_matrix(num_classes)
    weighted_loss_sum = 0.0
    valid_pixel_sum = 0
    amp_enabled = _mixed_precision_enabled(device, use_amp)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid_pixels = int((targets != ignore_index).sum().item())
        if valid_pixels == 0:
            continue
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Получен нечисловой validation loss: {loss.item()}")

        weighted_loss_sum += float(loss.item()) * valid_pixels
        valid_pixel_sum += valid_pixels
        update_confusion_matrix(confusion, logits, targets, ignore_index)

    if valid_pixel_sum == 0:
        raise ValueError("Dev-набор не содержит ни одного неигнорируемого пикселя")
    result = calculate_metrics(confusion)
    result["loss"] = weighted_loss_sum / valid_pixel_sum
    return result


def _checkpoint_data(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    model_name: str,
    best_miou: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": best_miou,
        "config": config,
    }


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str | Path,
    history_path: str | Path,
    config: dict[str, Any],
    use_amp: bool = True,
    num_classes: int = 19,
    ignore_index: int = 255,
    on_epoch_end: Callable[[dict[str, float], int], None] | None = None,
    resume_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Run all epochs, saving CSV after each epoch plus best/last checkpoints."""
    if epochs <= 0:
        raise ValueError("Число эпох должно быть положительным")
    checkpoint_directory = Path(checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_directory / f"{model_name}_best.pt"
    last_path = checkpoint_directory / f"{model_name}_last.pt"

    amp_enabled = _mixed_precision_enabled(device, use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_miou = -math.inf
    history_rows: list[dict[str, float]] = []
    start_epoch = 1

    if resume_path is not None and Path(resume_path).is_file():
        try:
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
        except TypeError:  # Compatibility with PyTorch versions before weights_only.
            checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("model_name") != model_name:
            raise ValueError(
                f"Checkpoint {resume_path} относится к модели "
                f"{checkpoint.get('model_name')}, а не {model_name}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        best_miou = float(checkpoint.get("best_miou", -math.inf))
        completed_epoch = int(checkpoint["epoch"])
        start_epoch = completed_epoch + 1
        if history_file.is_file():
            previous_history = pd.read_csv(history_file)
            previous_history = previous_history.loc[
                previous_history["epoch"] <= completed_epoch
            ]
            history_rows = previous_history.to_dict(orient="records")
        else:
            warnings.warn(
                f"Checkpoint найден, но история отсутствует: {history_file}. "
                "CSV продолжится с возобновлённой эпохи."
            )
        print(
            f"[{model_name}] resume из {resume_path}: "
            f"следующая эпоха {start_epoch}/{epochs}"
        )

    if start_epoch > epochs:
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Обучение уже завершено, но best checkpoint не найден: {best_path}"
            )
        print(f"[{model_name}] уже обучена до эпохи {epochs}, повторный запуск пропущен")
        return pd.DataFrame(history_rows), best_path, Path(resume_path or last_path)

    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()

        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            num_classes,
            ignore_index,
        )
        dev_result = validate(
            model,
            dev_loader,
            criterion,
            device,
            use_amp,
            num_classes,
            ignore_index,
        )
        epoch_seconds = time.perf_counter() - started_at
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        row: dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_result["loss"]),
            "dev_loss": float(dev_result["loss"]),
            "epoch_seconds": float(epoch_seconds),
            "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        }
        row.update(flatten_metrics(train_result, "train"))
        row.update(flatten_metrics(dev_result, "dev"))
        history_rows.append(row)
        history = pd.DataFrame(history_rows)
        history.to_csv(history_file, index=False, encoding="utf-8")

        current_miou = float(dev_result["miou"])
        if not math.isfinite(current_miou):
            raise RuntimeError("Dev mIoU не является конечным числом")
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(
                _checkpoint_data(
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    model_name,
                    best_miou,
                    config,
                ),
                best_path,
            )
        torch.save(
            _checkpoint_data(
                model,
                optimizer,
                scaler,
                epoch,
                model_name,
                best_miou,
                config,
            ),
            last_path,
        )

        if on_epoch_end is not None:
            on_epoch_end(row, epoch)
        print(
            f"[{model_name}] epoch {epoch:02d}/{epochs}: "
            f"train_loss={row['train_loss']:.4f}, "
            f"dev_loss={row['dev_loss']:.4f}, "
            f"dev_mIoU={row['dev_miou']:.4f}, "
            f"time={epoch_seconds:.1f}s, peak_gpu={peak_gpu_memory_mb:.0f}MB"
        )

    return pd.DataFrame(history_rows), best_path, last_path
