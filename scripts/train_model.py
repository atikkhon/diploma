"""Train or resume one independently configured segmentation model."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CityscapesDataset, build_transform  # noqa: E402
from src.experiment import load_run, make_run_paths  # noqa: E402
from src.models import create_model  # noqa: E402
from src.tracking import (  # noqa: E402
    configure_mlflow,
    finite_metrics,
    flatten_parameters,
    log_artifacts,
    read_run_id,
    write_run_id,
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
from src.visualization import save_training_curves  # noqa: E402


def create_loaders(
    config: dict[str, Any], project_root: Path, seed: int
) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    training = config["training"]
    common = {
        "manifest_path": resolve_path(data["split_file"], project_root),
        "dataset_root": resolve_path(data["root"], project_root),
        "width": int(data["image_width"]),
        "height": int(data["image_height"]),
    }
    augmentation = config.get("augmentation", {"policy": "baseline"})
    train_transform = build_transform(
        train=True,
        width=common["width"],
        height=common["height"],
        augmentation_config=augmentation,
    )
    train_dataset = CityscapesDataset(
        **common,
        split="train",
        train=True,
        transform=train_transform,
    )
    dev_dataset = CityscapesDataset(**common, split="dev", train=False)
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size должен быть > 0, num_workers должен быть >= 0")
    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
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
        f"DataLoader: train={len(train_dataset)}, dev={len(dev_dataset)}, "
        f"batch_size={batch_size}, workers={num_workers}, "
        f"augmentation={augmentation.get('policy', 'baseline')}",
        flush=True,
    )
    return train_loader, dev_loader


def write_run_config(config: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return path


def checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "epoch" not in checkpoint:
        raise ValueError(f"Checkpoint не содержит epoch: {path}")
    return int(checkpoint["epoch"])


def history_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    history = pd.read_csv(path)
    if "epoch" not in history.columns or history.empty:
        return 0
    return int(history["epoch"].max())


def sync_config_epochs(config: dict[str, Any], paths) -> None:
    observed_epoch = max(
        history_epoch(paths.history),
        checkpoint_epoch(paths.last_checkpoint) if paths.last_checkpoint.is_file() else 0,
        checkpoint_epoch(paths.best_checkpoint) if paths.best_checkpoint.is_file() else 0,
    )
    training = config["training"]
    configured_epochs = int(training["epochs"])
    if observed_epoch > configured_epochs:
        training["epochs"] = observed_epoch
        training["epochs_repaired_from_artifacts"] = True


def resolve_source_run(source: str, current_run_root: Path) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_absolute():
        return source_path.resolve()
    sibling = (current_run_root.parent / source_path).resolve()
    if sibling.is_dir():
        return sibling
    run_relative = (current_run_root.parent.parent / source_path).resolve()
    if run_relative.is_dir():
        return run_relative
    return (PROJECT_ROOT / source_path).resolve()


def source_model_directory(source_run: Path) -> Path:
    model_dir = source_run.parent.parent / "models" / source_run.name
    if model_dir.is_dir():
        return model_dir
    raise FileNotFoundError(
        "Не найден каталог модели исходного run. В новой структуре он должен быть здесь: "
        f"{model_dir}"
    )


def prepare_continuation(
    config: dict[str, Any],
    paths,
    source: str,
    checkpoint_kind: str,
) -> tuple[Path, Path]:
    if checkpoint_kind not in {"last", "best"}:
        raise ValueError("--init-checkpoint должен быть last или best")
    source_run = resolve_source_run(source, paths.root)
    source_models = source_model_directory(source_run)
    source_checkpoint = source_models / f"{checkpoint_kind}.pt"
    source_best = source_models / "best.pt"
    source_history = source_run / "metrics" / "training_history.csv"
    if not source_run.is_dir():
        raise FileNotFoundError(f"Исходный run не найден: {source_run}")
    if not source_history.is_file():
        raise FileNotFoundError(f"История исходного run не найдена: {source_history}")
    if not source_best.is_file():
        raise FileNotFoundError(f"Best checkpoint исходного run не найден: {source_best}")

    initial_epoch = checkpoint_epoch(source_checkpoint)
    additional_epochs = int(config["training"]["epochs"])
    if additional_epochs <= 0:
        raise ValueError("Для дообучения training.epochs должен быть числом новых эпох")
    config["training"]["epochs"] = initial_epoch + additional_epochs
    config["training"]["additional_epochs"] = additional_epochs
    config["training"]["initial_checkpoint_epoch"] = initial_epoch
    config["training"]["init_from_run"] = str(source_run)
    config["training"]["init_from_model_dir"] = str(source_models)
    config["training"]["init_checkpoint"] = str(source_checkpoint)
    config["training"]["init_checkpoint_kind"] = checkpoint_kind
    shutil.copy2(source_best, paths.best_checkpoint)
    return source_checkpoint, source_history


def run_training(
    config_path: str | Path,
    resume: bool = False,
    continue_from_run: str | None = None,
    init_checkpoint: str = "last",
) -> None:
    if resume and continue_from_run is not None:
        raise ValueError("Нельзя одновременно использовать --resume и --continue-from-run")
    config, project_root, paths = load_run(config_path)
    paths.create()
    saved_config = paths.root / "run_config.yaml"
    if resume and saved_config.is_file():
        config = load_yaml(saved_config)
        if "project_root" in config:
            project_root = resolve_path(config["project_root"], project_root)
        else:
            config["project_root"] = str(project_root)
        paths = make_run_paths(config, project_root)
        paths.create()
        saved_config = paths.root / "run_config.yaml"
    else:
        config["project_root"] = str(project_root)

    existing_state = [
        path
        for path in (
            paths.best_checkpoint,
            paths.last_checkpoint,
            paths.history,
            paths.run_id,
        )
        if path.is_file()
    ]
    if existing_state and not resume and continue_from_run is None:
        raise FileExistsError(
            "Запуск уже содержит результаты. Укажите --resume или задайте новый "
            f"run.name: {', '.join(map(str, existing_state))}"
        )
    if existing_state and continue_from_run is not None:
        raise FileExistsError(
            "Новый run для дообучения должен быть пустым. Задайте новый run.name: "
            f"{', '.join(map(str, existing_state))}"
        )
    if resume and not paths.last_checkpoint.is_file() and existing_state:
        raise FileNotFoundError(
            f"Для resume отсутствует last checkpoint: {paths.last_checkpoint}"
        )

    continue_checkpoint: Path | None = None
    continue_history: Path | None = None
    if continue_from_run is not None:
        continue_checkpoint, continue_history = prepare_continuation(
            config,
            paths,
            continue_from_run,
            init_checkpoint,
        )
    if not (resume and saved_config.is_file()):
        write_run_config(config, saved_config)

    seed = int(config.get("seed", 42))
    data = config["data"]
    model_settings = config["model"]
    training = config["training"]
    model_name = str(model_settings["name"]).lower()
    device = select_device(str(training.get("device", "auto")))
    seed_everything(seed)
    train_loader, dev_loader = create_loaders(config, project_root, seed)
    model = create_model(
        model_name,
        classes=int(data["num_classes"]),
        settings=model_settings,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=int(data["ignore_index"]))
    environment_path = save_json(
        environment_info(), paths.metrics / "training_environment.json"
    )

    experiment = configure_mlflow(str(config["tracking"]["experiment_name"]))
    resume_path = None
    resume_history_path = None
    if continue_checkpoint is not None:
        resume_path = continue_checkpoint
        resume_history_path = continue_history
    elif resume and paths.last_checkpoint.is_file():
        resume_path = paths.last_checkpoint
    if resume_path is not None and continue_checkpoint is None:
        active_run = mlflow.start_run(run_id=read_run_id(paths.run_id))
    else:
        tags = {"model": model_name, "stage": "training"}
        if continue_from_run is not None:
            tags["continued_from_run"] = str(
                resolve_source_run(continue_from_run, paths.root)
            )
        active_run = mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=str(config["run"]["name"]),
            tags=tags,
        )

    with active_run as run:
        if resume_path is None or continue_checkpoint is not None:
            write_run_id(paths.run_id, run.info.run_id)
            mlflow.log_params(flatten_parameters(config))
        print(f"MLflow run_id: {run.info.run_id}", flush=True)
        train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=int(training["epochs"]),
            checkpoint_dir=paths.checkpoints,
            history_path=paths.history,
            config=config,
            use_amp=bool(training.get("mixed_precision", True)),
            num_classes=int(data["num_classes"]),
            ignore_index=int(data["ignore_index"]),
            on_epoch_end=lambda row, epoch: mlflow.log_metrics(
                finite_metrics(row), step=epoch
            ),
            resume_path=resume_path,
            resume_history_path=resume_history_path,
        )
        sync_config_epochs(config, paths)
        write_run_config(config, saved_config)
        plot_paths = save_training_curves(paths.history, paths.figures)
        log_artifacts(
            [
                paths.history,
                environment_path,
                saved_config,
                *plot_paths,
            ],
            artifact_path="training",
        )
        mlflow.set_tag("status", "completed")

    print(f"Run: {config['run']['name']}")
    print(f"Best checkpoint: {paths.best_checkpoint}")
    print(f"Training CSV: {paths.history}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--continue-from-run",
        help="Создать новый run и дообучить модель из указанного предыдущего run",
    )
    parser.add_argument(
        "--init-checkpoint",
        choices=("last", "best"),
        default="last",
        help="Какой checkpoint предыдущего run использовать для дообучения",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        args.config,
        resume=args.resume,
        continue_from_run=args.continue_from_run,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
