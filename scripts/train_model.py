"""Train or resume one independently configured segmentation model."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CityscapesDataset  # noqa: E402
from src.experiment import load_run  # noqa: E402
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
    make_dataloader_generator,
    resolve_path,
    save_json,
    seed_everything,
    seed_worker,
    select_device,
)


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
    train_dataset = CityscapesDataset(**common, split="train", train=True)
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
        f"batch_size={batch_size}, workers={num_workers}",
        flush=True,
    )
    return train_loader, dev_loader


def run_training(config_path: str | Path, resume: bool = False) -> None:
    config, project_root, paths = load_run(config_path)
    paths.create()
    config_file = Path(config_path).expanduser().resolve()
    saved_config = paths.root / "run_config.yaml"
    if config_file != saved_config.resolve():
        shutil.copy2(config_file, saved_config)

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
    if existing_state and not resume:
        raise FileExistsError(
            "Запуск уже содержит результаты. Укажите --resume или задайте новый "
            f"run.name: {', '.join(map(str, existing_state))}"
        )
    if resume and not paths.last_checkpoint.is_file() and existing_state:
        raise FileNotFoundError(
            f"Для resume отсутствует last checkpoint: {paths.last_checkpoint}"
        )

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
    resume_path = paths.last_checkpoint if resume and paths.last_checkpoint.is_file() else None
    if resume_path is not None:
        active_run = mlflow.start_run(run_id=read_run_id(paths.run_id))
    else:
        active_run = mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=str(config["run"]["name"]),
            tags={"model": model_name, "stage": "training"},
        )

    with active_run as run:
        if resume_path is None:
            write_run_id(paths.run_id, run.info.run_id)
            mlflow.log_params(flatten_parameters(config))
        print(f"MLflow run_id: {run.info.run_id}", flush=True)
        _, best_path, last_path = train_model(
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
        )
        log_artifacts(
            [paths.history, best_path, last_path, environment_path, saved_config],
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
