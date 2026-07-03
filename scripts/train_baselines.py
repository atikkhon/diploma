"""Train U-Net, DeepLabV3+ and PSPNet sequentially on the internal split."""

import argparse
import sys
import warnings
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


BASELINE_MODELS = ["unet", "deeplabv3plus", "pspnet"]


def checkpoint_epoch(path: Path, expected_model: str) -> int:
    """Read and validate the completed epoch without constructing the model."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Некорректный checkpoint: {path}")
    if "epoch" not in checkpoint or "model_name" not in checkpoint:
        raise ValueError(f"В checkpoint нет epoch или model_name: {path}")
    saved_model = str(checkpoint["model_name"]).lower()
    if saved_model != expected_model.lower():
        raise ValueError(
            f"В {path} сохранена модель {saved_model}, ожидалась {expected_model}"
        )
    return int(checkpoint["epoch"])


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
        help=(
            "Продолжить каждую выбранную модель из <model>_last.pt. "
            "Если checkpoint отсутствует, начать с первой эпохи."
        ),
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
    train_dataset = CityscapesDataset(**common, split="train", train=True)
    dev_dataset = CityscapesDataset(**common, split="dev", train=False)
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
    loader_options["generator"] = make_dataloader_generator(seed + 1)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_options)
    print(
        f"DataLoader: train={len(train_dataset)} изображений/"
        f"{len(train_loader)} batch, dev={len(dev_dataset)} изображений/"
        f"{len(dev_loader)} batch, workers={num_workers}, batch_size={batch_size}",
        flush=True,
    )
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
        history_path = history_dir / f"training_history_{model_name}.csv"
        run_id_path = history_dir / f"mlflow_run_id_{model_name}.txt"
        resume_path = checkpoint_dir / f"{model_name}_last.pt"
        selected_resume_path: Path | None = None
        existing_run_id: str | None = None
        if resume:
            if resume_path.is_file():
                selected_resume_path = resume_path
                completed_epoch = checkpoint_epoch(resume_path, model_name)
                print(
                    f"[{model_name}] найден checkpoint для resume: {resume_path} "
                    f"(epoch {completed_epoch})",
                    flush=True,
                )
                if completed_epoch >= int(training["epochs"]):
                    best_path = checkpoint_dir / f"{model_name}_best.pt"
                    missing_outputs = [
                        path
                        for path in (best_path, history_path)
                        if not path.is_file()
                    ]
                    if missing_outputs:
                        raise FileNotFoundError(
                            "Last checkpoint завершён, но отсутствуют обязательные "
                            "результаты: "
                            + ", ".join(map(str, missing_outputs))
                        )
                    print(
                        f"[{model_name}] обучение уже завершено "
                        f"({completed_epoch}/{training['epochs']}); "
                        "повторный запуск не требуется.",
                        flush=True,
                    )
                    continue
                if run_id_path.is_file():
                    existing_run_id = run_id_path.read_text(
                        encoding="utf-8"
                    ).strip() or None
            else:
                print(
                    f"[{model_name}] checkpoint для resume не найден; "
                    "обучение начнётся с эпохи 1.",
                    flush=True,
                )

        print(f"[{model_name}] 1/4: фиксация seed и создание DataLoader...", flush=True)
        seed_everything(seed)
        train_loader, dev_loader = create_loaders(config, project_root, seed)
        print(
            f"[{model_name}] 2/4: создание модели "
            f"{model_config.get('encoder', 'resnet34')} "
            f"(веса {model_config.get('encoder_weights', 'imagenet')})...",
            flush=True,
        )
        model = create_model(
            model_name,
            classes=int(data["num_classes"]),
            encoder_name=str(model_config.get("encoder", "resnet34")),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
        ).to(device)
        print(f"[{model_name}] 3/4: модель загружена на {device}", flush=True)
        optimizer = AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss(ignore_index=int(data["ignore_index"]))
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
        run_name = f"baseline_{model_name}_seed{seed}"
        tags = {
            "model_name": model_name,
            "seed": str(seed),
            "source": "training",
            "status": "running",
        }
        print(f"[{model_name}] 4/4: запуск обучения...", flush=True)
        with mlflow_run(
            experiment_name,
            run_name,
            run_parameters,
            run_id_path=run_id_path,
            existing_run_id=existing_run_id,
            tags=tags,
        ) as mlflow_module:
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
                resume_path=selected_resume_path,
            )
            for artifact in (history_path, best_path, last_path, environment_path):
                log_artifact_safe(mlflow_module, artifact)
            if mlflow_module is not None:
                try:
                    mlflow_module.set_tag("status", "completed")
                except Exception as error:
                    warnings.warn(f"Не удалось установить MLflow status tag: {error}")

        del model, optimizer, criterion, train_loader, dev_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    try:
        train_baselines(args.config, args.models, resume=args.resume)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Ошибка обучения: {error}") from error


if __name__ == "__main__":
    main()
