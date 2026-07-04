"""Retrain the selected baseline architecture with only robust augmentation changed."""

import argparse
import copy
import gc
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_clean import MODEL_LABELS, MODEL_NAMES, load_checkpoint  # noqa: E402
from scripts.evaluate_corruptions import (  # noqa: E402
    append_condition_rows,
    build_robustness_summary,
    corruption_families,
    create_evaluation_loader,
    evaluate_condition,
    prepare_official_val_manifest,
    validate_complete_results,
)
from src.corruptions import (  # noqa: E402
    CORRUPTION_NAMES,
    SEVERITY_LEVELS,
    CorruptionTransform,
    load_corruption_config,
)
from src.dataset import CityscapesDataset  # noqa: E402
from src.models import create_model  # noqa: E402
from src.robust_augmentation import (  # noqa: E402
    DEFAULT_SEEN_CORRUPTIONS,
    DEFAULT_UNSEEN_CORRUPTIONS,
    RobustTrainingTransform,
    resolve_robust_policy,
)
from src.tracking import log_artifact_safe, log_metrics_safe, mlflow_run  # noqa: E402
from src.train import train_model  # noqa: E402
from src.utils import (  # noqa: E402
    environment_info,
    load_yaml,
    make_dataloader_generator,
    resolve_path,
    save_json,
    seed_everything,
    seed_worker,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--corruptions", default="configs/corruptions.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить из robust_<model>_last.pt; без файла начать с epoch 1.",
    )
    return parser.parse_args()


def select_model_from_summary(summary_path: str | Path) -> str:
    """Read the single rank-1 architecture from robustness_summary.csv."""
    path = Path(summary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"robustness_summary.csv не найден: {path}. "
            "Сначала выполните evaluate_corruptions.py."
        )
    summary = pd.read_csv(path)
    required = {
        "model",
        "mean_corrupted_miou",
        "clean_miou",
        "worst_case_miou",
        "peak_gpu_memory_mb",
        "robustness_rank",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"В robustness_summary отсутствуют столбцы: {sorted(missing)}")
    winners = summary.loc[pd.to_numeric(summary["robustness_rank"]) == 1]
    if len(winners) != 1:
        raise ValueError("robustness_summary должен содержать ровно одну модель rank=1")
    model_name = str(winners.iloc[0]["model"]).lower()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"В robustness_summary выбрана неизвестная модель: {model_name}")
    return model_name


def _nested(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"В checkpoint config отсутствует {path}")
        value = value[key]
    return value


def validate_same_hyperparameters(
    current_config: Mapping[str, Any],
    baseline_checkpoint: Mapping[str, Any],
) -> None:
    """Prove that robust retraining changes no baseline hyperparameter."""
    baseline_config = baseline_checkpoint.get("config")
    if not isinstance(baseline_config, Mapping):
        raise ValueError("Baseline checkpoint не содержит config для сверки параметров")
    fixed_paths = (
        "seed",
        "data.split_file",
        "data.num_classes",
        "data.ignore_index",
        "data.image_width",
        "data.image_height",
        "models.encoder",
        "models.encoder_weights",
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "training.weight_decay",
        "training.mixed_precision",
    )
    mismatches = []
    for path in fixed_paths:
        baseline_value = _nested(baseline_config, path)
        current_value = _nested(current_config, path)
        if baseline_value != current_value:
            mismatches.append(
                f"{path}: baseline={baseline_value}, robust={current_value}"
            )
    if mismatches:
        raise ValueError(
            "Robust training может изменять только augmentation policy: "
            + "; ".join(mismatches)
        )


def validate_fixed_protocol(config: Mapping[str, Any]) -> None:
    """Reject changes to the thesis protocol before starting GPU work."""
    fixed = {
        "seed": 42,
        "data.num_classes": 19,
        "data.ignore_index": 255,
        "data.image_width": 384,
        "data.image_height": 192,
        "models.encoder": "resnet34",
        "models.encoder_weights": "imagenet",
        "training.epochs": 8,
        "training.learning_rate": 0.0003,
        "training.weight_decay": 0.0001,
        "training.mixed_precision": True,
    }
    wrong = [
        f"{path}={_nested(config, path)} (ожидалось {expected})"
        for path, expected in fixed.items()
        if _nested(config, path) != expected
    ]
    if wrong:
        raise ValueError("Нарушен зафиксированный протокол: " + "; ".join(wrong))
    if int(_nested(config, "training.batch_size")) <= 0:
        raise ValueError("training.batch_size должен быть положительным")


def validate_training_paths(dataset: CityscapesDataset, split: str) -> None:
    """Ensure internal train/dev contains only paths from official train."""
    for column in ("image_path", "mask_path"):
        for value in dataset.rows[column].astype(str):
            parts = {part.lower() for part in Path(value).parts}
            if "val" in parts or "train" not in parts:
                raise ValueError(
                    f"В split={split} найден путь не из official train: {value}"
                )


def create_robust_loaders(
    config: dict[str, Any],
    project_root: Path,
    policy: Mapping[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Use the original split and loader settings with only train transform changed."""
    data = config["data"]
    training = config["training"]
    seed = int(config["seed"])
    dataset_root = resolve_path(data["root"], project_root)
    manifest_path = resolve_path(data["split_file"], project_root)
    train_transform = RobustTrainingTransform(
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        policy=policy,
    )
    train_dataset = CityscapesDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="train",
        train=True,
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        transform=train_transform,
    )
    dev_dataset = CityscapesDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="dev",
        train=False,
        width=int(data["image_width"]),
        height=int(data["image_height"]),
    )
    validate_training_paths(train_dataset, "train")
    validate_training_paths(dev_dataset, "dev")
    num_workers = int(training.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers должен быть неотрицательным")
    options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=make_dataloader_generator(seed),
        **options,
    )
    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        generator=make_dataloader_generator(seed + 1),
        **options,
    )
    print(
        f"Robust DataLoader: train={len(train_dataset)}/{len(train_loader)} batch, "
        f"dev={len(dev_dataset)}/{len(dev_loader)} batch, "
        f"batch_size={training['batch_size']}, workers={num_workers}",
        flush=True,
    )
    return train_loader, dev_loader


def resolve_seen_unseen(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Read fixed groups, supporting runtime YAML created before these fields existed."""
    robust = config.get("robust_training", {})
    seen = list(robust.get("seen_corruptions", DEFAULT_SEEN_CORRUPTIONS))
    unseen = list(robust.get("unseen_corruptions", DEFAULT_UNSEEN_CORRUPTIONS))
    if set(seen) & set(unseen):
        raise ValueError("seen_corruptions и unseen_corruptions пересекаются")
    if set(seen) | set(unseen) != set(CORRUPTION_NAMES):
        raise ValueError("Seen/unseen группы должны покрывать ровно восемь corruptions")
    return seen, unseen


def evaluate_robust_model(
    config: dict[str, Any],
    project_root: Path,
    corruption_config: Mapping[str, Any],
    architecture: str,
    robust_model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate the robust best checkpoint on clean plus all 24 corrupted conditions."""
    checkpoint = load_checkpoint(checkpoint_path)
    if str(checkpoint.get("model_name")).lower() != robust_model_name:
        raise ValueError(
            f"В {checkpoint_path} записано {checkpoint.get('model_name')}, "
            f"ожидалось {robust_model_name}"
        )
    model = create_model(
        architecture,
        classes=19,
        encoder_name=str(config["models"]["encoder"]),
        encoder_weights=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    evaluation = config["evaluation"]
    training = config["training"]
    seed = int(config["seed"])
    result_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    family_by_corruption = corruption_families(corruption_config)

    with tempfile.TemporaryDirectory() as temporary:
        manifest_path, dataset_root = prepare_official_val_manifest(
            config,
            project_root,
            Path(temporary) / "official_val_manifest.csv",
        )
        _, clean_loader = create_evaluation_loader(
            manifest_path,
            dataset_root,
            config["data"],
            int(evaluation["batch_size"]),
            int(training.get("num_workers", 0)),
            device,
            seed,
        )
        clean_metrics, clean_resources = evaluate_condition(
            model,
            clean_loader,
            device,
            num_classes=19,
            ignore_index=255,
            use_amp=bool(training["mixed_precision"]),
            progress_label=f"{robust_model_name}/clean",
        )
        clean_miou = float(clean_metrics["miou"])
        append_condition_rows(
            result_rows,
            per_class_rows,
            robust_model_name,
            checkpoint_path,
            int(checkpoint["epoch"]),
            "clean",
            "clean",
            0,
            clean_metrics,
            clean_resources,
            clean_miou,
        )
        del clean_loader

        condition_number = 1
        for corruption in CORRUPTION_NAMES:
            for severity in SEVERITY_LEVELS:
                condition_number += 1
                transform = CorruptionTransform(
                    corruption, severity, corruption_config
                )
                _, loader = create_evaluation_loader(
                    manifest_path,
                    dataset_root,
                    config["data"],
                    int(evaluation["batch_size"]),
                    int(training.get("num_workers", 0)),
                    device,
                    seed,
                    image_corruption=transform,
                )
                print(
                    f"[{robust_model_name}] evaluation {condition_number}/25: "
                    f"{corruption} severity={severity}",
                    flush=True,
                )
                metrics, resources = evaluate_condition(
                    model,
                    loader,
                    device,
                    num_classes=19,
                    ignore_index=255,
                    use_amp=bool(training["mixed_precision"]),
                    progress_label=f"{robust_model_name}/{corruption}/S{severity}",
                )
                append_condition_rows(
                    result_rows,
                    per_class_rows,
                    robust_model_name,
                    checkpoint_path,
                    int(checkpoint["epoch"]),
                    corruption,
                    family_by_corruption[corruption],
                    severity,
                    metrics,
                    resources,
                    clean_miou,
                )
                del loader, transform, metrics, resources
                gc.collect()
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    robust_results = pd.DataFrame(result_rows)
    if len(robust_results) != 25:
        raise ValueError(f"Robust evaluation содержит {len(robust_results)} условий вместо 25")
    return robust_results


def build_robust_comparison(
    baseline_results: pd.DataFrame,
    robust_results: pd.DataFrame,
    architecture: str,
) -> pd.DataFrame:
    """Create one condition-aligned baseline-versus-robust table."""
    baseline = baseline_results.loc[
        baseline_results["model"] == architecture
    ].copy()
    if len(baseline) != 25 or len(robust_results) != 25:
        raise ValueError("Для comparison нужны 25 baseline и 25 robust условий")
    keys = ["corruption", "family", "severity"]
    baseline_columns = keys + [
        "miou",
        "macro_dice",
        "pixel_accuracy",
        "delta_miou",
        "retention",
        "mean_inference_ms_per_image",
        "checkpoint",
    ]
    robust_columns = baseline_columns
    merged = baseline[baseline_columns].merge(
        robust_results[robust_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_robust"),
    )
    if len(merged) != 25:
        raise ValueError(f"После merge осталось {len(merged)} условий вместо 25")
    merged.insert(0, "architecture", architecture)
    merged["miou_change_robust_minus_baseline"] = (
        merged["miou_robust"] - merged["miou_baseline"]
    )
    merged["macro_dice_change_robust_minus_baseline"] = (
        merged["macro_dice_robust"] - merged["macro_dice_baseline"]
    )
    merged["pixel_accuracy_change_robust_minus_baseline"] = (
        merged["pixel_accuracy_robust"] - merged["pixel_accuracy_baseline"]
    )
    corruption_order = {"clean": -1, **{name: i for i, name in enumerate(CORRUPTION_NAMES)}}
    merged["_order"] = merged["corruption"].map(corruption_order)
    return merged.sort_values(["_order", "severity"]).drop(columns="_order").reset_index(drop=True)


def build_seen_unseen_comparison(
    comparison: pd.DataFrame,
    seen: list[str],
    unseen: list[str],
) -> pd.DataFrame:
    """Aggregate baseline and robust metrics separately for fixed seen/unseen groups."""
    rows = []
    for group_name, corruptions in (("seen", seen), ("unseen", unseen)):
        group = comparison.loc[comparison["corruption"].isin(corruptions)]
        expected = len(corruptions) * len(SEVERITY_LEVELS)
        if len(group) != expected:
            raise ValueError(
                f"Группа {group_name} содержит {len(group)} условий вместо {expected}"
            )
        rows.append(
            {
                "architecture": str(comparison.iloc[0]["architecture"]),
                "group": group_name,
                "corruptions": ",".join(corruptions),
                "condition_count": len(group),
                "baseline_mean_miou": float(group["miou_baseline"].mean()),
                "robust_mean_miou": float(group["miou_robust"].mean()),
                "miou_change_robust_minus_baseline": float(
                    group["miou_robust"].mean() - group["miou_baseline"].mean()
                ),
                "baseline_mean_retention": float(group["retention_baseline"].mean()),
                "robust_mean_retention": float(group["retention_robust"].mean()),
                "retention_change_robust_minus_baseline": float(
                    group["retention_robust"].mean()
                    - group["retention_baseline"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_baseline_vs_robust(comparison: pd.DataFrame, destination: Path) -> None:
    rows = []
    clean = comparison.loc[comparison["corruption"] == "clean"].iloc[0]
    rows.append(("clean", clean["miou_baseline"], clean["miou_robust"]))
    for corruption in CORRUPTION_NAMES:
        group = comparison.loc[comparison["corruption"] == corruption]
        rows.append(
            (
                corruption,
                float(group["miou_baseline"].mean()),
                float(group["miou_robust"].mean()),
            )
        )
    labels, baseline_values, robust_values = zip(*rows)
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(positions - width / 2, baseline_values, width, label="Baseline")
    axis.bar(positions + width / 2, robust_values, width, label="Robust")
    axis.set_xticks(positions, labels, rotation=40, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("mIoU")
    axis.set_title("Baseline vs robust: clean and mean over severity")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_seen_unseen(summary: pd.DataFrame, destination: Path) -> None:
    positions = np.arange(len(summary))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, baseline_column, robust_column, ylabel in (
        (
            axes[0],
            "baseline_mean_miou",
            "robust_mean_miou",
            "Mean mIoU",
        ),
        (
            axes[1],
            "baseline_mean_retention",
            "robust_mean_retention",
            "Mean retention",
        ),
    ):
        axis.bar(positions - width / 2, summary[baseline_column], width, label="Baseline")
        axis.bar(positions + width / 2, summary[robust_column], width, label="Robust")
        axis.set_xticks(positions, summary["group"])
        axis.set_ylim(0.0, max(1.0, float(summary[robust_column].max()) * 1.08))
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.3)
    axes[0].legend()
    figure.suptitle("Seen and unseen corruption comparison")
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_clean_tradeoff(comparison: pd.DataFrame, destination: Path) -> None:
    clean = comparison.loc[comparison["corruption"] == "clean"].iloc[0]
    labels = ["mIoU", "Macro Dice", "Pixel accuracy"]
    baseline_values = [
        clean["miou_baseline"],
        clean["macro_dice_baseline"],
        clean["pixel_accuracy_baseline"],
    ]
    robust_values = [
        clean["miou_robust"],
        clean["macro_dice_robust"],
        clean["pixel_accuracy_robust"],
    ]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(positions - width / 2, baseline_values, width, label="Baseline")
    axis.bar(positions + width / 2, robust_values, width, label="Robust")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value on clean official val")
    axis.set_title("Clean-quality trade-off")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_robust(
    config_path: str | Path,
    corruption_config_path: str | Path,
    resume: bool = False,
) -> dict[str, Path]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    for section in ("data", "models", "training", "evaluation"):
        if section not in config:
            raise ValueError(f"В конфигурации отсутствует раздел {section}")
    validate_fixed_protocol(config)
    project_root = config_file.parent.parent
    training = config["training"]
    evaluation = config["evaluation"]
    robust_config = config.get("robust_training", {})
    if not isinstance(robust_config, Mapping):
        raise ValueError("robust_training должен быть словарём")
    if robust_config.get("enabled", True) is not True:
        raise ValueError("robust_training.enabled должен быть true")

    metrics_dir = resolve_path(evaluation["metrics_dir"], project_root)
    history_dir = resolve_path(
        training.get("history_dir", evaluation["metrics_dir"]), project_root
    )
    figures_dir = resolve_path(
        evaluation.get("figures_dir", "outputs/figures"), project_root
    )
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selection_path = resolve_path(
        robust_config.get(
            "selection_file", metrics_dir / "robustness_summary.csv"
        ),
        project_root,
    )
    architecture = select_model_from_summary(selection_path)
    baseline_checkpoint_path = checkpoint_dir / f"{architecture}_best.pt"
    baseline_checkpoint = load_checkpoint(baseline_checkpoint_path)
    if str(baseline_checkpoint.get("model_name")).lower() != architecture:
        raise ValueError(f"Некорректный baseline checkpoint: {baseline_checkpoint_path}")
    validate_same_hyperparameters(config, baseline_checkpoint)

    corruption_file = resolve_path(corruption_config_path, project_root)
    corruption_config = load_corruption_config(corruption_file)
    family_by_corruption = corruption_families(corruption_config)
    families = list(dict.fromkeys(family_by_corruption.values()))
    baseline_results_path = metrics_dir / "corruption_results.csv"
    if not baseline_results_path.is_file():
        raise FileNotFoundError(
            f"Baseline corruption results не найдены: {baseline_results_path}"
        )
    baseline_results = pd.read_csv(baseline_results_path)
    validate_complete_results(baseline_results)
    if "checkpoint_epoch" not in baseline_results.columns:
        raise ValueError("В corruption_results.csv отсутствует checkpoint_epoch")
    rebuilt_summary = build_robustness_summary(baseline_results, families)
    rebuilt_winner = str(
        rebuilt_summary.loc[rebuilt_summary["robustness_rank"] == 1, "model"].iloc[0]
    )
    if rebuilt_winner != architecture:
        raise ValueError(
            "robustness_summary.csv не соответствует corruption_results.csv: "
            f"rank=1 summary={architecture}, пересчёт={rebuilt_winner}"
        )
    selected_baseline_rows = baseline_results.loc[
        baseline_results["model"] == architecture
    ]
    saved_epochs = set(selected_baseline_rows["checkpoint_epoch"].astype(int))
    baseline_checkpoint_epoch = int(baseline_checkpoint["epoch"])
    if saved_epochs != {baseline_checkpoint_epoch}:
        raise ValueError(
            "corruption_results.csv рассчитан для другой версии baseline checkpoint"
        )
    del baseline_checkpoint
    seen, unseen = resolve_seen_unseen(config)
    policy = resolve_robust_policy(config)
    robust_model_name = f"robust_{architecture}"
    history_path = history_dir / "robust_training_history.csv"
    run_id_path = history_dir / f"mlflow_run_id_{robust_model_name}.txt"
    robust_last_path = checkpoint_dir / f"{robust_model_name}_last.pt"
    resume_path = robust_last_path if resume and robust_last_path.is_file() else None
    if resume and resume_path is None:
        print("Robust last checkpoint не найден; обучение начнётся с epoch 1.", flush=True)
    existing_run_id = None
    if resume_path is not None and run_id_path.is_file():
        existing_run_id = run_id_path.read_text(encoding="utf-8").strip() or None

    run_config = copy.deepcopy(config)
    run_config.setdefault("robust_training", {})["selected_model"] = architecture
    run_config["robust_training"]["augmentation"] = policy
    run_config["robust_training"]["seen_corruptions"] = seen
    run_config["robust_training"]["unseen_corruptions"] = unseen
    device = select_device(str(training.get("device", "auto")))
    seed_everything(int(config["seed"]))
    environment_path = save_json(
        environment_info(), history_dir / "robust_training_environment.json"
    )
    print(
        f"Robust architecture из robustness_summary.csv: {architecture}; device={device}",
        flush=True,
    )
    print(
        "Изменяется только train augmentation; optimizer, LR, split, seed, "
        "epochs, batch size и checkpoint criterion совпадают с baseline.",
        flush=True,
    )

    train_loader, dev_loader = create_robust_loaders(
        config, project_root, policy, device
    )
    model = create_model(
        architecture,
        classes=19,
        encoder_name=str(config["models"]["encoder"]),
        encoder_weights=str(config["models"]["encoder_weights"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    output_paths = {
        "robust_training_history": history_path,
        "robust_comparison": metrics_dir / "robust_comparison.csv",
        "seen_unseen_comparison": metrics_dir / "seen_unseen_comparison.csv",
        "baseline_vs_robust": figures_dir / "baseline_vs_robust.png",
        "seen_unseen": figures_dir / "seen_unseen.png",
        "clean_robust_tradeoff": figures_dir / "clean_robust_tradeoff.png",
    }
    for name, path in output_paths.items():
        if name != "robust_training_history" and path.is_file():
            path.unlink()

    experiment_name = config.get("tracking", {}).get(
        "experiment_name", "cityscapes_robustness"
    )
    with mlflow_run(
        experiment_name,
        f"{robust_model_name}_seed{config['seed']}",
        run_config,
        run_id_path=run_id_path,
        existing_run_id=existing_run_id,
        tags={
            "model_name": architecture,
            "variant": "robust",
            "seed": str(config["seed"]),
            "source": "robust_training",
            "status": "running",
        },
    ) as mlflow_module:
        _, robust_best_path, robust_last_path = train_model(
            model=model,
            model_name=robust_model_name,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=int(training["epochs"]),
            checkpoint_dir=checkpoint_dir,
            history_path=history_path,
            config=run_config,
            use_amp=bool(training["mixed_precision"]),
            num_classes=19,
            ignore_index=255,
            on_epoch_end=lambda row, epoch: log_metrics_safe(
                mlflow_module, row, epoch
            ),
            resume_path=resume_path,
        )
        del model, optimizer, criterion, train_loader, dev_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        robust_results = evaluate_robust_model(
            config,
            project_root,
            corruption_config,
            architecture,
            robust_model_name,
            robust_best_path,
            device,
        )
        comparison = build_robust_comparison(
            baseline_results, robust_results, architecture
        )
        comparison.to_csv(
            output_paths["robust_comparison"], index=False, encoding="utf-8"
        )
        comparison_from_csv = pd.read_csv(output_paths["robust_comparison"])
        seen_unseen_summary = build_seen_unseen_comparison(
            comparison_from_csv, seen, unseen
        )
        seen_unseen_summary.to_csv(
            output_paths["seen_unseen_comparison"],
            index=False,
            encoding="utf-8",
        )
        seen_unseen_from_csv = pd.read_csv(
            output_paths["seen_unseen_comparison"]
        )
        plot_baseline_vs_robust(
            comparison_from_csv, output_paths["baseline_vs_robust"]
        )
        plot_seen_unseen(seen_unseen_from_csv, output_paths["seen_unseen"])
        plot_clean_tradeoff(
            comparison_from_csv, output_paths["clean_robust_tradeoff"]
        )
        for artifact in (
            history_path,
            robust_best_path,
            robust_last_path,
            environment_path,
            *output_paths.values(),
        ):
            log_artifact_safe(mlflow_module, artifact)
        if mlflow_module is not None:
            try:
                mlflow_module.set_tag("status", "completed")
            except Exception as error:
                print(f"Предупреждение MLflow status: {error}", flush=True)

    print(
        f"Robust training и evaluation завершены для {MODEL_LABELS[architecture]}",
        flush=True,
    )
    print(f"robust_best_checkpoint: {robust_best_path}")
    print(f"robust_last_checkpoint: {robust_last_path}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return {**output_paths, "robust_best_checkpoint": robust_best_path}


def main() -> None:
    args = parse_args()
    try:
        train_robust(args.config, args.corruptions, resume=args.resume)
    except (
        KeyError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        raise SystemExit(f"Ошибка robust training: {error}") from error


if __name__ == "__main__":
    main()
