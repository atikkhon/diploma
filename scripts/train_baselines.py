"""Train U-Net, DeepLabV3+ and PSPNet sequentially on the internal split."""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CityscapesDataset  # noqa: E402
from src.models import create_model  # noqa: E402
from src.tracking import (  # noqa: E402
    log_artifact_safe,
    log_metrics_safe,
    mlflow_run,
)
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


BASELINE_MODELS = ["unet", "deeplabv3plus", "pspnet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        help="Обучить только указанные модели; по умолчанию обучаются все",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить каждую модель из <model>_last.pt, если он существует",
    )
    return parser.parse_args()


def require_sections(config: dict[str, Any]) -> None:
    missing = [name for name in ("data", "models", "training") if name not in config]
    if missing:
        raise ValueError(f"В конфигурации отсутствуют разделы: {missing}")
    data = config["data"]
    training = config["training"]
    fixed_values = {
        "seed": (config.get("seed"), 42),
        "data.num_classes": (data.get("num_classes"), 19),
        "data.ignore_index": (data.get("ignore_index"), 255),
        "models.encoder": (config["models"].get("encoder"), "resnet34"),
        "models.encoder_weights": (
            config["models"].get("encoder_weights"),
            "imagenet",
        ),
        "training.epochs": (training.get("epochs"), 8),
        "training.learning_rate": (training.get("learning_rate"), 0.0003),
        "training.weight_decay": (training.get("weight_decay"), 0.0001),
        "training.mixed_precision": (training.get("mixed_precision"), True),
    }
    wrong = [
        f"{name}={actual} (ожидалось {expected})"
        for name, (actual, expected) in fixed_values.items()
        if actual != expected
    ]
    if wrong:
        raise ValueError("Нарушены зафиксированные параметры: " + "; ".join(wrong))


def create_loaders(
    config: dict[str, Any], project_root: Path, seed: int
) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    training = config["training"]
    dataset_root = resolve_path(data["root"], project_root)
    manifest_path = resolve_path(data["split_file"], project_root)
    common = {
        "manifest_path": manifest_path,
        "dataset_root": dataset_root,
        "width": int(data["image_width"]),
        "height": int(data["image_height"]),
    }
    train_dataset = CityscapesDataset(
        **common, split="train", train=True
    )
    dev_dataset = CityscapesDataset(
        **common, split="dev", train=False
    )
    for split_name, dataset in (("train", train_dataset), ("dev", dev_dataset)):
        for column in ("image_path", "mask_path"):
            for value in dataset.rows[column].astype(str):
                parts = {part.lower() for part in Path(value).parts}
                if "val" in parts or "train" not in parts:
                    raise ValueError(
                        f"В split={split_name} найден путь не из официального train: "
                        f"{value}. Официальный val нельзя использовать для выбора checkpoint."
                    )

    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size должен быть > 0, num_workers должен быть >= 0")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": make_dataloader_generator(seed),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    # A separate generator prevents train iteration from changing dev state.
    loader_options["generator"] = make_dataloader_generator(seed + 1)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_options)
    return train_loader, dev_loader


def train_baselines(
    config_path: str | Path,
    model_names: list[str] | None = None,
    resume: bool = False,
) -> None:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    require_sections(config)
    project_root = config_file.parent.parent
    seed = int(config["seed"])
    training = config["training"]
    data = config["data"]
    model_config = config["models"]
    device = select_device(str(training.get("device", "auto")))

    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    history_dir = resolve_path(
        training.get("history_dir", "outputs/metrics"), project_root
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    environment_path = save_json(
        environment_info(), history_dir / "training_environment.json"
    )
    print(f"Устройство: {device}")
    print(f"Информация о среде: {environment_path}")

    configured_names = [str(name).lower() for name in model_config.get("names", [])]
    if configured_names != BASELINE_MODELS:
        raise ValueError(
            "models.names должен содержать в указанном порядке: "
            + ", ".join(BASELINE_MODELS)
        )

    selected_models = model_names or BASELINE_MODELS
    for model_name in selected_models:
        seed_everything(seed)
        train_loader, dev_loader = create_loaders(config, project_root, seed)
        model = create_model(
            model_name,
            classes=int(data["num_classes"]),
            encoder_name=str(model_config.get("encoder", "resnet34")),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss(ignore_index=int(data["ignore_index"]))
        history_path = history_dir / f"training_history_{model_name}.csv"
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        run_parameters = {
            **config,
            "run": {
                "model_name": model_name,
                "device": str(device),
                "parameter_count": parameter_count,
                "mixed_precision_active": bool(
                    training.get("mixed_precision", True) and device.type == "cuda"
                ),
            },
        }

        experiment_name = config.get("tracking", {}).get(
            "experiment_name", "cityscapes_robustness"
        )
        with mlflow_run(experiment_name, model_name, run_parameters) as mlflow_module:
            resume_path = checkpoint_dir / f"{model_name}_last.pt"
            _, best_path, last_path = train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                dev_loader=dev_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=int(training["epochs"]),
                checkpoint_dir=checkpoint_dir,
                history_path=history_path,
                config=config,
                use_amp=bool(training.get("mixed_precision", True)),
                num_classes=int(data["num_classes"]),
                ignore_index=int(data["ignore_index"]),
                on_epoch_end=lambda row, epoch: log_metrics_safe(
                    mlflow_module, row, epoch
                ),
                resume_path=resume_path if resume else None,
            )
            for artifact in (history_path, best_path, last_path, environment_path):
                log_artifact_safe(mlflow_module, artifact)

        del model, optimizer, criterion, train_loader, dev_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    try:
        train_baselines(args.config, args.models, args.resume)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка обучения: {error}") from error


if __name__ == "__main__":
    main()
