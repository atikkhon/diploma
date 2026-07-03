"""Evaluate three best checkpoints on clean and corrupted official Cityscapes val."""

import argparse
import gc
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_clean import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    load_checkpoint,
    require_official_val_path,
)
from src.corruptions import (  # noqa: E402
    CORRUPTION_NAMES,
    SEVERITY_LEVELS,
    CorruptionTransform,
    create_corruption_manifest,
    load_corruption_config,
)
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


EXPECTED_CONDITION_COUNT = 1 + len(CORRUPTION_NAMES) * len(SEVERITY_LEVELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--corruptions", default="configs/corruptions.yaml")
    return parser.parse_args()


def corruption_families(config: Mapping[str, Any]) -> dict[str, str]:
    """Return corruption-to-family mapping in fixed YAML order."""
    result: dict[str, str] = {}
    for corruption in CORRUPTION_NAMES:
        family = config["corruptions"][corruption].get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"Для {corruption} не указано семейство")
        result[corruption] = family.strip()
    return result


def prepare_official_val_manifest(
    config: dict[str, Any],
    project_root: Path,
    destination: Path,
) -> tuple[Path, Path]:
    """Create one temporary clean manifest for exactly 500 official val pairs."""
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
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination, dataset_root


def create_evaluation_loader(
    manifest_path: Path,
    dataset_root: Path,
    data_config: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    image_corruption: CorruptionTransform | None = None,
) -> tuple[CityscapesDataset, DataLoader]:
    """Create a deterministic loader; corrupted RGB is generated per sample."""
    dataset = CityscapesDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="val",
        train=False,
        width=int(data_config["image_width"]),
        height=int(data_config["image_height"]),
        image_corruption=image_corruption,
    )
    if len(dataset) != 500:
        raise ValueError(f"Evaluation dataset содержит {len(dataset)} вместо 500")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
        # A new condition gets a new dataset, so workers must stop after each pass.
        persistent_workers=False,
    )
    return dataset, loader


@torch.inference_mode()
def evaluate_condition(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    use_amp: bool,
    progress_label: str,
) -> tuple[dict[str, float | list[float]], dict[str, float]]:
    """Accumulate one confusion matrix over all 500 images, then calculate metrics."""
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
                f"[{progress_label}] batch {batch_index}/{batch_count}",
                flush=True,
            )

    if image_count != 500:
        raise ValueError(f"Обработано {image_count} изображений вместо 500")
    if confusion.sum().item() == 0:
        raise ValueError(f"Confusion matrix пуста: {progress_label}")
    metrics = calculate_metrics(confusion)
    for name in ("miou", "macro_dice", "pixel_accuracy"):
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"Метрика {name} не является числом: {progress_label}")
    resources = {
        "total_inference_seconds": total_inference_seconds,
        "mean_inference_ms_per_image": (
            total_inference_seconds * 1000.0 / image_count
        ),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    return metrics, resources


def append_condition_rows(
    result_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    model_name: str,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    corruption: str,
    family: str,
    severity: int,
    metrics: Mapping[str, Any],
    resources: Mapping[str, float],
    clean_miou: float,
) -> None:
    """Append summary and 19 class rows using a fixed clean-reference formula."""
    miou = float(metrics["miou"])
    if clean_miou <= 0.0:
        raise ValueError(
            f"Clean mIoU модели {model_name} равен {clean_miou}; retention не определён"
        )
    is_clean = corruption == "clean"
    result_rows.append(
        {
            "model": model_name,
            "corruption": corruption,
            "family": family,
            "severity": severity,
            "num_images": 500,
            "miou": miou,
            "macro_dice": float(metrics["macro_dice"]),
            "pixel_accuracy": float(metrics["pixel_accuracy"]),
            # Positive value means loss relative to clean; improvement is negative.
            "delta_miou": 0.0 if is_clean else clean_miou - miou,
            "retention": 1.0 if is_clean else miou / clean_miou,
            "total_inference_seconds": float(resources["total_inference_seconds"]),
            "mean_inference_ms_per_image": float(
                resources["mean_inference_ms_per_image"]
            ),
            "peak_gpu_memory_mb": float(resources["peak_gpu_memory_mb"]),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
        }
    )
    iou_values = metrics["iou_per_class"]
    if not isinstance(iou_values, list) or len(iou_values) != 19:
        raise ValueError(f"Для {model_name}/{corruption}/S{severity} получено не 19 IoU")
    for class_id, (class_name, iou) in enumerate(
        zip(CITYSCAPES_CLASS_NAMES, iou_values)
    ):
        per_class_rows.append(
            {
                "model": model_name,
                "corruption": corruption,
                "family": family,
                "severity": severity,
                "class_id": class_id,
                "class_name": class_name,
                "iou": float(iou),
            }
        )


def validate_complete_results(results: pd.DataFrame) -> None:
    """Reject partial or duplicated evaluation tables before reporting."""
    required = {
        "model",
        "corruption",
        "family",
        "severity",
        "miou",
        "macro_dice",
        "delta_miou",
        "retention",
        "total_inference_seconds",
        "mean_inference_ms_per_image",
        "peak_gpu_memory_mb",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"В corruption_results отсутствуют столбцы: {sorted(missing)}")
    duplicate_columns = ["model", "corruption", "severity"]
    if results.duplicated(duplicate_columns).any():
        raise ValueError("corruption_results содержит повторяющиеся условия")
    for model_name in MODEL_NAMES:
        model_rows = results.loc[results["model"] == model_name]
        if len(model_rows) != EXPECTED_CONDITION_COUNT:
            raise ValueError(
                f"Для {model_name} должно быть {EXPECTED_CONDITION_COUNT} условий, "
                f"получено {len(model_rows)}"
            )
        clean_rows = model_rows.loc[model_rows["corruption"] == "clean"]
        if len(clean_rows) != 1 or int(clean_rows.iloc[0]["severity"]) != 0:
            raise ValueError(f"Для {model_name} отсутствует единственная clean-строка")
        for corruption in CORRUPTION_NAMES:
            severities = set(
                model_rows.loc[
                    model_rows["corruption"] == corruption, "severity"
                ].astype(int)
            )
            if severities != set(SEVERITY_LEVELS):
                raise ValueError(
                    f"Для {model_name}/{corruption} отсутствуют severity 1, 2 или 3"
                )


def validate_complete_per_class(per_class: pd.DataFrame) -> None:
    """Require exactly 19 unique class rows for every model and condition."""
    required = {
        "model",
        "corruption",
        "family",
        "severity",
        "class_id",
        "class_name",
        "iou",
    }
    missing = required - set(per_class.columns)
    if missing:
        raise ValueError(
            f"В corruption_per_class отсутствуют столбцы: {sorted(missing)}"
        )
    key = ["model", "corruption", "severity", "class_id"]
    if per_class.duplicated(key).any():
        raise ValueError("corruption_per_class содержит повторяющиеся class rows")
    expected_rows = (
        len(MODEL_NAMES) * EXPECTED_CONDITION_COUNT * len(CITYSCAPES_CLASS_NAMES)
    )
    if len(per_class) != expected_rows:
        raise ValueError(
            f"corruption_per_class содержит {len(per_class)} строк вместо "
            f"{expected_rows}"
        )
    expected_ids = set(range(len(CITYSCAPES_CLASS_NAMES)))
    grouped = per_class.groupby(["model", "corruption", "severity"])
    for condition, rows in grouped:
        if set(rows["class_id"].astype(int)) != expected_ids:
            raise ValueError(f"Для условия {condition} записаны не все 19 классов")


def build_robustness_summary(
    results: pd.DataFrame,
    families: list[str],
) -> pd.DataFrame:
    """Aggregate robustness per model and apply the fixed lexicographic ranking."""
    validate_complete_results(results)
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model_rows = results.loc[results["model"] == model_name]
        clean = model_rows.loc[model_rows["corruption"] == "clean"].iloc[0]
        corrupted = model_rows.loc[model_rows["corruption"] != "clean"]
        row: dict[str, Any] = {
            "model": model_name,
            "clean_miou": float(clean["miou"]),
            "mean_corrupted_miou": float(corrupted["miou"].mean()),
            "mean_retention": float(corrupted["retention"].mean()),
            "worst_case_miou": float(corrupted["miou"].min()),
            "peak_gpu_memory_mb": float(model_rows["peak_gpu_memory_mb"].max()),
        }
        for family in families:
            family_rows = corrupted.loc[corrupted["family"] == family]
            if family_rows.empty:
                raise ValueError(f"Для семейства {family} нет результатов")
            row[f"family_{family}_miou"] = float(family_rows["miou"].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_indices = summary.sort_values(
        by=[
            "mean_corrupted_miou",
            "clean_miou",
            "worst_case_miou",
            "peak_gpu_memory_mb",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).index.tolist()
    if len(ordered_indices) > 1:
        selection_columns = [
            "mean_corrupted_miou",
            "clean_miou",
            "worst_case_miou",
            "peak_gpu_memory_mb",
        ]
        first = summary.loc[ordered_indices[0], selection_columns].tolist()
        second = summary.loc[ordered_indices[1], selection_columns].tolist()
        if first == second:
            tied_models = summary.loc[ordered_indices[:2], "model"].tolist()
            raise ValueError(
                "Правила выбора не разрешают полную ничью моделей: "
                + ", ".join(tied_models)
            )
    rank_by_index = {index: rank for rank, index in enumerate(ordered_indices, 1)}
    summary["robustness_rank"] = [rank_by_index[index] for index in summary.index]
    summary["is_best_model"] = summary["robustness_rank"] == 1
    summary["selection_rule"] = (
        "mean_corrupted_miou desc; clean_miou desc; worst_case_miou desc; "
        "peak_gpu_memory_mb asc"
    )
    return summary.sort_values("robustness_rank").reset_index(drop=True)


def plot_robustness_heatmap(results: pd.DataFrame, destination: Path) -> None:
    corrupted = results.loc[results["corruption"] != "clean"].copy()
    condition_order = [
        (corruption, severity)
        for corruption in CORRUPTION_NAMES
        for severity in SEVERITY_LEVELS
    ]
    values = np.array(
        [
            [
                corrupted.loc[
                    (corrupted["model"] == model_name)
                    & (corrupted["corruption"] == corruption)
                    & (corrupted["severity"] == severity),
                    "miou",
                ].iloc[0]
                for corruption, severity in condition_order
            ]
            for model_name in MODEL_NAMES
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(19, 4.5), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_yticks(range(len(MODEL_NAMES)), [MODEL_LABELS[name] for name in MODEL_NAMES])
    axis.set_xticks(
        range(len(condition_order)),
        [f"{name}\nS{severity}" for name, severity in condition_order],
        rotation=55,
        ha="right",
    )
    axis.set_title("Corrupted mIoU on official Cityscapes validation")
    figure.colorbar(image, ax=axis, label="mIoU")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_degradation_curves(results: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for axis, corruption in zip(axes.ravel(), CORRUPTION_NAMES):
        for model_name in MODEL_NAMES:
            rows = results.loc[
                (results["model"] == model_name)
                & (results["corruption"] == corruption)
            ].sort_values("severity")
            clean_miou = float(
                results.loc[
                    (results["model"] == model_name)
                    & (results["corruption"] == "clean"),
                    "miou",
                ].iloc[0]
            )
            axis.plot(
                [0, *rows["severity"].astype(int).tolist()],
                [clean_miou, *rows["miou"].astype(float).tolist()],
                marker="o",
                label=MODEL_LABELS[model_name],
            )
        axis.set_title(corruption)
        axis.set_xticks([0, *SEVERITY_LEVELS], ["clean", "1", "2", "3"])
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Severity")
        axis.set_ylabel("mIoU")
        axis.grid(alpha=0.3)
    axes[0, 0].legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_retention_comparison(summary: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(
        [MODEL_LABELS[name] for name in summary["model"]],
        summary["mean_retention"],
        color="tab:blue",
    )
    axis.set_ylim(0.0, max(1.0, float(summary["mean_retention"].max()) * 1.08))
    axis.set_ylabel("Mean retention")
    axis.set_title("Mean corrupted-to-clean mIoU retention")
    axis.grid(axis="y", alpha=0.3)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_family_comparison(
    summary: pd.DataFrame,
    families: list[str],
    destination: Path,
) -> None:
    positions = np.arange(len(families))
    width = 0.25
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for model_index, model_name in enumerate(MODEL_NAMES):
        row = summary.loc[summary["model"] == model_name].iloc[0]
        axis.bar(
            positions + (model_index - 1) * width,
            [row[f"family_{family}_miou"] for family in families],
            width,
            label=MODEL_LABELS[model_name],
        )
    axis.set_xticks(positions, families)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean corrupted mIoU")
    axis.set_title("Robustness by corruption family")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_worst_case_comparison(summary: pd.DataFrame, destination: Path) -> None:
    positions = np.arange(len(summary))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, column, label in (
        (-1, "clean_miou", "Clean mIoU"),
        (0, "mean_corrupted_miou", "Mean corrupted mIoU"),
        (1, "worst_case_miou", "Worst-case mIoU"),
    ):
        axis.bar(positions + offset * width, summary[column], width, label=label)
    axis.set_xticks(positions, [MODEL_LABELS[name] for name in summary["model"]])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("mIoU")
    axis.set_title("Clean, mean corrupted and worst-case comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_corruptions(
    config_path: str | Path,
    corruption_config_path: str | Path,
) -> dict[str, Path]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    for section in ("data", "models", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"В конфигурации отсутствует раздел {section}")
    project_root = config_file.parent.parent
    corruption_file = resolve_path(corruption_config_path, project_root)
    corruption_config = load_corruption_config(corruption_file)
    family_by_corruption = corruption_families(corruption_config)
    families = list(dict.fromkeys(family_by_corruption.values()))

    data = config["data"]
    model_config = config["models"]
    training = config["training"]
    evaluation = config["evaluation"]
    if [str(name).lower() for name in model_config.get("names", [])] != MODEL_NAMES:
        raise ValueError("models.names должен содержать unet, deeplabv3plus, pspnet")
    if int(data["num_classes"]) != 19 or int(data["ignore_index"]) != 255:
        raise ValueError(
            "Corruption evaluation зафиксирован для 19 классов и ignore_index=255"
        )
    batch_size = int(evaluation["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("evaluation.batch_size должен быть > 0, num_workers >= 0")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = select_device(str(training.get("device", "auto")))
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    metrics_dir = resolve_path(evaluation["metrics_dir"], project_root)
    figures_dir = resolve_path(
        evaluation.get("figures_dir", "outputs/figures"), project_root
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        model_name: checkpoint_dir / f"{model_name}_best.pt"
        for model_name in MODEL_NAMES
    }
    missing_checkpoints = [
        path for path in checkpoint_paths.values() if not path.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Не найдены best checkpoints: "
            + ", ".join(map(str, missing_checkpoints))
        )

    output_paths = {
        "corruption_results": metrics_dir / "corruption_results.csv",
        "corruption_per_class": metrics_dir / "corruption_per_class.csv",
        "robustness_summary": metrics_dir / "robustness_summary.csv",
        "corruption_manifest": metrics_dir / "corruption_manifest.csv",
        "robustness_heatmap": figures_dir / "robustness_heatmap.png",
        "degradation_curves": figures_dir / "degradation_curves.png",
        "retention_comparison": figures_dir / "retention_comparison.png",
        "corruption_family_comparison": (
            figures_dir / "corruption_family_comparison.png"
        ),
        "worst_case_comparison": figures_dir / "worst_case_comparison.png",
    }
    # Never leave previous complete reports beside a new partial evaluation.
    for output_path in output_paths.values():
        if output_path.is_file():
            output_path.unlink()
    result_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as temporary:
        clean_manifest, dataset_root = prepare_official_val_manifest(
            config,
            project_root,
            Path(temporary) / "official_val_manifest.csv",
        )
        create_corruption_manifest(
            clean_manifest,
            output_paths["corruption_manifest"],
            corruption_config,
            split="val",
        )
        print(
            "Corruption evaluation: 3 модели × 25 условий × "
            f"500 изображений, device={device}, batch_size={batch_size}",
            flush=True,
        )

        for model_name in MODEL_NAMES:
            checkpoint_path = checkpoint_paths[model_name]
            checkpoint = load_checkpoint(checkpoint_path)
            if str(checkpoint.get("model_name")).lower() != model_name:
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
            checkpoint_epoch = int(checkpoint["epoch"])

            _, clean_loader = create_evaluation_loader(
                clean_manifest,
                dataset_root,
                data,
                batch_size,
                num_workers,
                device,
                seed,
            )
            clean_metrics, clean_resources = evaluate_condition(
                model,
                clean_loader,
                device,
                num_classes=19,
                ignore_index=255,
                use_amp=bool(training.get("mixed_precision", True)),
                progress_label=f"{model_name}/clean",
            )
            clean_miou = float(clean_metrics["miou"])
            append_condition_rows(
                result_rows,
                per_class_rows,
                model_name,
                checkpoint_path,
                checkpoint_epoch,
                "clean",
                "clean",
                0,
                clean_metrics,
                clean_resources,
                clean_miou,
            )
            pd.DataFrame(result_rows).to_csv(
                output_paths["corruption_results"], index=False, encoding="utf-8"
            )
            pd.DataFrame(per_class_rows).to_csv(
                output_paths["corruption_per_class"], index=False, encoding="utf-8"
            )
            del clean_loader

            condition_number = 1
            for corruption in CORRUPTION_NAMES:
                for severity in SEVERITY_LEVELS:
                    condition_number += 1
                    transform = CorruptionTransform(
                        corruption, severity, corruption_config
                    )
                    _, dataloader = create_evaluation_loader(
                        clean_manifest,
                        dataset_root,
                        data,
                        batch_size,
                        num_workers,
                        device,
                        seed,
                        image_corruption=transform,
                    )
                    label = f"{model_name}/{corruption}/S{severity}"
                    print(
                        f"[{model_name}] условие {condition_number}/"
                        f"{EXPECTED_CONDITION_COUNT}: {corruption} severity={severity}",
                        flush=True,
                    )
                    metrics, resources = evaluate_condition(
                        model,
                        dataloader,
                        device,
                        num_classes=19,
                        ignore_index=255,
                        use_amp=bool(training.get("mixed_precision", True)),
                        progress_label=label,
                    )
                    append_condition_rows(
                        result_rows,
                        per_class_rows,
                        model_name,
                        checkpoint_path,
                        checkpoint_epoch,
                        corruption,
                        family_by_corruption[corruption],
                        severity,
                        metrics,
                        resources,
                        clean_miou,
                    )
                    # Preserve completed conditions if Colab is interrupted.
                    pd.DataFrame(result_rows).to_csv(
                        output_paths["corruption_results"],
                        index=False,
                        encoding="utf-8",
                    )
                    pd.DataFrame(per_class_rows).to_csv(
                        output_paths["corruption_per_class"],
                        index=False,
                        encoding="utf-8",
                    )
                    del dataloader, transform, metrics, resources
                    gc.collect()
            del model, checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Every aggregate, ranking, plot and printed conclusion is rebuilt from CSV.
    results_from_csv = pd.read_csv(output_paths["corruption_results"])
    validate_complete_results(results_from_csv)
    per_class_from_csv = pd.read_csv(output_paths["corruption_per_class"])
    validate_complete_per_class(per_class_from_csv)
    summary = build_robustness_summary(results_from_csv, families)
    summary.to_csv(
        output_paths["robustness_summary"], index=False, encoding="utf-8"
    )

    summary_from_csv = pd.read_csv(output_paths["robustness_summary"])
    plot_robustness_heatmap(
        results_from_csv, output_paths["robustness_heatmap"]
    )
    plot_degradation_curves(
        results_from_csv, output_paths["degradation_curves"]
    )
    plot_retention_comparison(
        summary_from_csv, output_paths["retention_comparison"]
    )
    plot_family_comparison(
        summary_from_csv,
        families,
        output_paths["corruption_family_comparison"],
    )
    plot_worst_case_comparison(
        summary_from_csv, output_paths["worst_case_comparison"]
    )

    best_rows = summary_from_csv.loc[summary_from_csv["robustness_rank"] == 1]
    if len(best_rows) != 1:
        raise ValueError("В robustness_summary должна быть ровно одна лучшая модель")
    best_model = str(best_rows.iloc[0]["model"])
    print(
        "Лучшая модель по зафиксированным правилам robustness_summary.csv: "
        f"{best_model}",
        flush=True,
    )
    print(
        "Порядок выбора: mean corrupted mIoU, clean mIoU, worst-case mIoU, "
        "затем меньшая peak GPU memory.",
        flush=True,
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return output_paths


def main() -> None:
    args = parse_args()
    try:
        evaluate_corruptions(args.config, args.corruptions)
    except (
        KeyError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        raise SystemExit(f"Ошибка corruption evaluation: {error}") from error


if __name__ == "__main__":
    main()
