"""Train and validate segmentation models with global dataset metrics."""

import math
import time
from pathlib import Path
from typing import Any

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


def create_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """Create a GradScaler with the current PyTorch API."""
    return torch.amp.GradScaler(device.type, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Any,
    use_amp: bool = True,
    num_classes: int = 19,
    ignore_index: int = 255,
    log_interval: int = 25,
    progress_prefix: str = "train",
) -> dict[str, float | list[float]]:
    """Train for one epoch and report loss without costly train-set IoU."""
    model.train()
    del num_classes
    weighted_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_pixel_sum = 0
    amp_enabled = _mixed_precision_enabled(device, use_amp)
    batch_count = len(dataloader)
    started_at = time.perf_counter()
    print(
        f"{progress_prefix}: ожидание первого batch "
        f"(всего batch: {batch_count})...",
        flush=True,
    )

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        target_cpu = batch["mask"]
        valid_pixels = int((target_cpu != ignore_index).sum().item())
        if valid_pixels == 0:
            continue
        targets = target_cpu.to(device, non_blocking=True)

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

        weighted_loss_sum += loss.detach().to(torch.float64) * valid_pixels
        valid_pixel_sum += valid_pixels
        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == batch_count:
            elapsed = time.perf_counter() - started_at
            print(
                f"{progress_prefix}: batch {batch_index}/{batch_count}, "
                f"loss={loss.item():.4f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    if valid_pixel_sum == 0:
        raise ValueError("Train-набор не содержит ни одного неигнорируемого пикселя")
    return {"loss": float((weighted_loss_sum / valid_pixel_sum).item())}


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    num_classes: int = 19,
    ignore_index: int = 255,
    log_interval: int = 25,
    progress_prefix: str = "dev",
) -> dict[str, float | list[float]]:
    """Evaluate the complete set before calculating any segmentation metric."""
    model.eval()
    confusion = create_confusion_matrix(num_classes, device=device)
    weighted_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    valid_pixel_sum = 0
    amp_enabled = _mixed_precision_enabled(device, use_amp)
    batch_count = len(dataloader)
    started_at = time.perf_counter()
    print(
        f"{progress_prefix}: ожидание первого batch "
        f"(всего batch: {batch_count})...",
        flush=True,
    )

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        target_cpu = batch["mask"]
        valid_pixels = int((target_cpu != ignore_index).sum().item())
        if valid_pixels == 0:
            continue
        targets = target_cpu.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Получен нечисловой validation loss: {loss.item()}")

        weighted_loss_sum += loss.detach().to(torch.float64) * valid_pixels
        valid_pixel_sum += valid_pixels
        update_confusion_matrix(
            confusion,
            logits,
            targets,
            ignore_index,
            validate_indices=False,
        )
        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == batch_count:
            elapsed = time.perf_counter() - started_at
            print(
                f"{progress_prefix}: batch {batch_index}/{batch_count}, "
                f"loss={loss.item():.4f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    if valid_pixel_sum == 0:
        raise ValueError("Dev-набор не содержит ни одного неигнорируемого пикселя")
    result = calculate_metrics(confusion)
    result["loss"] = float((weighted_loss_sum / valid_pixel_sum).item())
    return result


def _checkpoint_data(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
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
    resume_path: str | Path | None = None,
    resume_history_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Run or resume training, saving CSV and best/last checkpoints."""
    if epochs <= 0:
        raise ValueError("Число эпох должно быть положительным")
    checkpoint_directory = Path(checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_directory / "best.pt"
    last_path = checkpoint_directory / "last.pt"

    amp_enabled = _mixed_precision_enabled(device, use_amp)
    scaler = create_grad_scaler(device, amp_enabled)
    best_miou = -math.inf
    history_rows: list[dict[str, float]] = []
    log_interval = int(config.get("training", {}).get("log_interval", 25))
    if log_interval <= 0:
        raise ValueError("training.log_interval должен быть положительным")

    start_epoch = 1
    resume_file = Path(resume_path) if resume_path is not None else None
    if resume_file is not None:
        if not resume_file.is_file():
            raise FileNotFoundError(f"Checkpoint для resume не найден: {resume_file}")
        checkpoint = torch.load(
            resume_file, map_location=device, weights_only=False
        )
        required_keys = {
            "epoch",
            "model_name",
            "model_state_dict",
            "optimizer_state_dict",
        }
        missing_keys = required_keys - set(checkpoint)
        if missing_keys:
            raise ValueError(
                f"Checkpoint {resume_file} не содержит поля: {sorted(missing_keys)}"
            )
        if str(checkpoint["model_name"]).lower() != model_name.lower():
            raise ValueError(
                f"Checkpoint предназначен для {checkpoint['model_name']}, "
                f"а запрошена модель {model_name}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        best_miou = float(checkpoint.get("best_miou", -math.inf))
        completed_epoch = int(checkpoint["epoch"])
        if completed_epoch < 0:
            raise ValueError(
                f"В checkpoint указана неверная эпоха: {completed_epoch}"
            )
        start_epoch = completed_epoch + 1

        history_source = (
            Path(resume_history_path)
            if resume_history_path is not None
            else history_file
        )
        if history_source.is_file():
            previous_history = pd.read_csv(history_source)
            if "epoch" not in previous_history.columns:
                raise ValueError(
                    f"В истории обучения нет столбца epoch: {history_source}"
                )
            previous_history = previous_history.loc[
                previous_history["epoch"] <= completed_epoch
            ]
            history_rows = previous_history.to_dict(orient="records")
        else:
            raise FileNotFoundError(
                f"Для resume необходим CSV истории: {history_source}"
            )
        print(
            f"[{model_name}] resume из {resume_file}: "
            f"завершена эпоха {completed_epoch}, следующая {start_epoch}",
            flush=True,
        )

    if start_epoch > epochs:
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Обучение дошло до эпохи {epochs}, но best checkpoint отсутствует: "
                f"{best_path}"
            )
        print(
            f"[{model_name}] уже завершено: checkpoint содержит эпоху "
            f"{start_epoch - 1}/{epochs}. Повторное обучение не требуется.",
            flush=True,
        )
        return pd.DataFrame(history_rows), best_path, resume_file or last_path

    for epoch in range(start_epoch, epochs + 1):
        train_sampler = getattr(train_loader, "sampler", None)
        set_epoch = getattr(train_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        print(
            f"[{model_name}] начало эпохи {epoch:02d}/{epochs}",
            flush=True,
        )

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
            log_interval,
            f"[{model_name}] train {epoch:02d}/{epochs}",
        )
        dev_result = validate(
            model,
            dev_loader,
            criterion,
            device,
            use_amp,
            num_classes,
            ignore_index,
            log_interval,
            f"[{model_name}] dev {epoch:02d}/{epochs}",
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

        print(
            f"[{model_name}] epoch {epoch:02d}/{epochs}: "
            f"train_loss={row['train_loss']:.4f}, "
            f"dev_loss={row['dev_loss']:.4f}, "
            f"dev_mIoU={row['dev_miou']:.4f}, "
            f"time={epoch_seconds:.1f}s, peak_gpu={peak_gpu_memory_mb:.0f}MB"
            , flush=True
        )

    return pd.DataFrame(history_rows), best_path, last_path
