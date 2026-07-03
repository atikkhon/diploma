"""Evaluate all best baseline checkpoints on the official Cityscapes val set."""

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CityscapesDataset, find_cityscapes_pairs  # noqa: E402
from src.metrics import (  # noqa: E402
    CITYSCAPES_CLASS_NAMES,
    calculate_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.models import create_model  # noqa: E402
from src.utils import (  # noqa: E402
    load_yaml,
    make_dataloader_generator,
    resolve_path,
    seed_everything,
    seed_worker,
    select_device,
)


MODEL_NAMES = ["unet", "deeplabv3plus", "pspnet"]
MODEL_LABELS = {
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "pspnet": "PSPNet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint не найден: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required_keys = {"model_state_dict", "model_name", "epoch"}
    if not isinstance(checkpoint, dict) or not required_keys <= set(checkpoint):
        raise ValueError(f"Некорректный checkpoint: {path}")
    return checkpoint


def require_official_val_path(value: str | Path, field_name: str) -> None:
    parts = {part.lower() for part in Path(value).parts}
    if "val" not in parts or "train" in parts:
        raise ValueError(
            f"{field_name} должен указывать на официальный val, получено: {value}"
        )


def preflight_artifacts(
    checkpoint_dir: Path,
    history_dir: Path,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Fail before evaluation if a required checkpoint or training CSV is missing."""
    checkpoint_paths: dict[str, Path] = {}
    histories: dict[str, pd.DataFrame] = {}
    required_history_columns = {"epoch", "train_loss", "dev_loss", "dev_miou"}
    for model_name in MODEL_NAMES:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Best checkpoint не найден: {checkpoint_path}")
        checkpoint_paths[model_name] = checkpoint_path

        history_path = history_dir / f"training_history_{model_name}.csv"
        if not history_path.is_file():
            raise FileNotFoundError(f"CSV истории обучения не найден: {history_path}")
        history = pd.read_csv(history_path)
        missing = required_history_columns - set(history.columns)
        if missing:
            raise ValueError(
                f"В {history_path} отсутствуют столбцы: {sorted(missing)}"
            )
        if history.empty:
            raise ValueError(f"CSV истории обучения пуст: {history_path}")
        histories[model_name] = history
    return checkpoint_paths, histories


def create_official_val_dataset(
    config: dict[str, Any],
    project_root: Path,
    temporary_directory: Path,
) -> CityscapesDataset:
    data = config["data"]
    require_official_val_path(
        data["official_val_images"], "data.official_val_images"
    )
    require_official_val_path(
        data["official_val_masks"], "data.official_val_masks"
    )
    dataset_root = resolve_path(data["root"], project_root)
    pairs = find_cityscapes_pairs(
        dataset_root,
        data["official_val_images"],
        data["official_val_masks"],
    )
    if len(pairs) != 500:
        raise ValueError(
            f"Официальный Cityscapes val должен содержать 500 пар, найдено: {len(pairs)}"
        )
    frame = pd.DataFrame(pairs)
    frame["split"] = "val"
    manifest_path = temporary_directory / "official_val_manifest.csv"
    frame.to_csv(manifest_path, index=False, encoding="utf-8")
    return CityscapesDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="val",
        train=False,
        width=int(data["image_width"]),
        height=int(data["image_height"]),
    )


@torch.inference_mode()
def evaluate_checkpoint(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    use_amp: bool,
    model_name: str,
) -> tuple[dict[str, float | list[float]], torch.Tensor, dict[str, float]]:
    """Run one deterministic pass and calculate metrics from one global matrix."""
    model.eval()
    confusion = create_confusion_matrix(num_classes, device=device)
    amp_enabled = use_amp and device.type == "cuda"
    total_inference_seconds = 0.0
    image_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_count = len(dataloader)
    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_inference_seconds += time.perf_counter() - started_at
        image_count += int(images.shape[0])
        update_confusion_matrix(
            confusion,
            logits,
            targets,
            ignore_index=ignore_index,
            validate_indices=False,
        )
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == batch_count:
            print(
                f"[{model_name}] clean batch {batch_index}/{batch_count}",
                flush=True,
            )

    if image_count != 500:
        raise ValueError(
            f"Во время оценки обработано {image_count} изображений вместо 500"
        )
    if confusion.sum().item() == 0:
        raise ValueError(f"Confusion matrix модели {model_name} пуста")
    metrics = calculate_metrics(confusion)
    if not math.isfinite(float(metrics["miou"])):
        raise ValueError(f"Clean mIoU модели {model_name} не является числом")
    resources = {
        "mean_inference_ms_per_image": (
            total_inference_seconds * 1000.0 / image_count
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    return metrics, confusion.detach().cpu(), resources


def save_confusion_matrix(
    confusion: torch.Tensor,
    model_name: str,
    metrics_dir: Path,
) -> Path:
    frame = pd.DataFrame(
        confusion.numpy(),
        index=CITYSCAPES_CLASS_NAMES,
        columns=CITYSCAPES_CLASS_NAMES,
    )
    destination = metrics_dir / f"confusion_matrix_{model_name}.csv"
    frame.to_csv(destination, index_label="target_class", encoding="utf-8")
    return destination


def plot_training_curves(
    histories: dict[str, pd.DataFrame],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for column, model_name in enumerate(MODEL_NAMES):
        history = histories[model_name]
        axes[0, column].plot(
            history["epoch"], history["train_loss"], marker="o", label="train loss"
        )
        axes[0, column].plot(
            history["epoch"], history["dev_loss"], marker="o", label="dev loss"
        )
        axes[0, column].set_title(MODEL_LABELS[model_name])
        axes[0, column].set_xlabel("Epoch")
        axes[0, column].set_ylabel("Cross-entropy loss")
        axes[0, column].grid(alpha=0.3)
        axes[0, column].legend()

        axes[1, column].plot(
            history["epoch"], history["dev_miou"], marker="o", color="tab:green"
        )
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel("Internal dev mIoU")
        axes[1, column].set_ylim(0.0, 1.0)
        axes[1, column].grid(alpha=0.3)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_clean_comparison(summary: pd.DataFrame, destination: Path) -> None:
    metric_columns = ["miou", "macro_dice", "pixel_accuracy"]
    labels = ["mIoU", "Macro Dice", "Pixel accuracy"]
    positions = np.arange(len(summary))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, (column, label) in enumerate(zip(metric_columns, labels)):
        axis.bar(positions + (index - 1) * width, summary[column], width, label=label)
    axis.set_xticks(positions, [MODEL_LABELS[name] for name in summary["model"]])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value")
    axis.set_title("Clean official validation comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_iou(per_class: pd.DataFrame, destination: Path) -> None:
    positions = np.arange(len(CITYSCAPES_CLASS_NAMES))
    width = 0.26
    figure, axis = plt.subplots(figsize=(18, 7), constrained_layout=True)
    for index, model_name in enumerate(MODEL_NAMES):
        values = per_class.loc[
            per_class["model"] == model_name
        ].sort_values("class_id")["iou"]
        axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
        )
    axis.set_xticks(positions, CITYSCAPES_CLASS_NAMES, rotation=55, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("IoU")
    axis.set_title("Per-class IoU on clean official validation")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_clean(config_path: str | Path) -> dict[str, Path]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    for section in ("data", "models", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"В конфигурации отсутствует раздел {section}")
    project_root = config_file.parent.parent
    data = config["data"]
    model_config = config["models"]
    training = config["training"]
    evaluation = config["evaluation"]
    if [str(name).lower() for name in model_config.get("names", [])] != MODEL_NAMES:
        raise ValueError("models.names должен содержать unet, deeplabv3plus, pspnet")
    if int(data["num_classes"]) != 19 or int(data["ignore_index"]) != 255:
        raise ValueError("Clean evaluation зафиксирован для 19 классов и ignore_index=255")
    if int(evaluation["batch_size"]) <= 0:
        raise ValueError("evaluation.batch_size должен быть положительным")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers должен быть неотрицательным")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(training.get("device", "auto")))
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    history_dir = resolve_path(
        training.get("history_dir", "outputs/metrics"), project_root
    )
    metrics_dir = resolve_path(evaluation["metrics_dir"], project_root)
    figures_dir = resolve_path(
        evaluation.get("figures_dir", "outputs/figures"), project_root
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths, histories = preflight_artifacts(checkpoint_dir, history_dir)

    summary_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    output_paths: dict[str, Path] = {}

    with tempfile.TemporaryDirectory() as temporary:
        dataset = create_official_val_dataset(
            config, project_root, Path(temporary)
        )
        num_workers = int(training.get("num_workers", 0))
        loader_options = {
            "batch_size": int(evaluation["batch_size"]),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "worker_init_fn": seed_worker,
            "generator": make_dataloader_generator(seed),
            "persistent_workers": num_workers > 0,
        }
        dataloader = DataLoader(dataset, **loader_options)
        print(
            f"Official clean val: {len(dataset)} изображений, "
            f"device={device}, batch_size={loader_options['batch_size']}",
            flush=True,
        )

        for model_name in MODEL_NAMES:
            checkpoint_path = checkpoint_paths[model_name]
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint.get("model_name") != model_name:
                raise ValueError(
                    f"В {checkpoint_path} записана модель "
                    f"{checkpoint.get('model_name')}, ожидалась {model_name}"
                )
            model = create_model(
                model_name,
                classes=19,
                encoder_name=str(model_config["encoder"]),
                encoder_weights=None,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            metrics, confusion, resources = evaluate_checkpoint(
                model,
                dataloader,
                device,
                num_classes=19,
                ignore_index=255,
                use_amp=bool(training.get("mixed_precision", True)),
                model_name=model_name,
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "evaluation_split": "official_cityscapes_val",
                    "selection_scope": "descriptive_final_evaluation_only",
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "num_images": len(dataset),
                    "miou": float(metrics["miou"]),
                    "macro_dice": float(metrics["macro_dice"]),
                    "pixel_accuracy": float(metrics["pixel_accuracy"]),
                    "mean_inference_ms_per_image": resources[
                        "mean_inference_ms_per_image"
                    ],
                    "num_parameters": parameter_count,
                    "peak_gpu_memory_mb": resources["peak_gpu_memory_mb"],
                }
            )
            resource_rows.append(
                {
                    "model": model_name,
                    "num_parameters": parameter_count,
                    "mean_inference_ms_per_image": resources[
                        "mean_inference_ms_per_image"
                    ],
                    "peak_gpu_memory_mb": resources["peak_gpu_memory_mb"],
                    "device": str(device),
                }
            )
            iou_values = metrics["iou_per_class"]
            for class_id, (class_name, iou) in enumerate(
                zip(CITYSCAPES_CLASS_NAMES, iou_values)
            ):
                per_class_rows.append(
                    {
                        "model": model_name,
                        "class_id": class_id,
                        "class_name": class_name,
                        "iou": float(iou),
                    }
                )
            output_paths[f"confusion_{model_name}"] = save_confusion_matrix(
                confusion, model_name, metrics_dir
            )
            print(
                f"[{model_name}] clean evaluation завершена; "
                "числовые результаты сохранены из фактической confusion matrix",
                flush=True,
            )
            del model, checkpoint, confusion
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    best_index = summary["miou"].idxmax()
    summary["clean_rank"] = summary["miou"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary["is_preliminary_best"] = False
    summary.loc[best_index, "is_preliminary_best"] = True
    preliminary_best = str(summary.loc[best_index, "model"])
    per_class = pd.DataFrame(per_class_rows)
    resources = pd.DataFrame(resource_rows)

    output_paths["clean_summary"] = metrics_dir / "clean_summary.csv"
    output_paths["clean_per_class_iou"] = metrics_dir / "clean_per_class_iou.csv"
    output_paths["resource_summary"] = metrics_dir / "resource_summary.csv"
    summary.to_csv(output_paths["clean_summary"], index=False, encoding="utf-8")
    per_class.to_csv(
        output_paths["clean_per_class_iou"], index=False, encoding="utf-8"
    )
    resources.to_csv(
        output_paths["resource_summary"], index=False, encoding="utf-8"
    )

    output_paths["training_curves"] = figures_dir / "training_curves.png"
    output_paths["clean_model_comparison"] = (
        figures_dir / "clean_model_comparison.png"
    )
    output_paths["per_class_clean_iou"] = figures_dir / "per_class_clean_iou.png"
    plot_training_curves(histories, output_paths["training_curves"])
    plot_clean_comparison(summary, output_paths["clean_model_comparison"])
    plot_per_class_iou(per_class, output_paths["per_class_clean_iou"])

    print(f"Предварительно лучшая модель по clean mIoU: {preliminary_best}")
    print(
        "Это описательный результат финальной оценки: скрипт не изменяет "
        "learning rate, число эпох, checkpoints или experiment.yaml."
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return output_paths


def main() -> None:
    args = parse_args()
    try:
        evaluate_clean(args.config)
    except (KeyError, FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка clean evaluation: {error}") from error


if __name__ == "__main__":
    main()
