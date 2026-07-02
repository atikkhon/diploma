"""Provide configuration, reproducibility, device and environment helpers."""

import json
import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch, including all visible CUDA devices."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give a deterministic NumPy/Python seed to one DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int = 42) -> torch.Generator:
    """Create the seeded generator passed to a PyTorch DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping and report missing or malformed files clearly."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"YAML-конфигурация не найдена: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Некорректный YAML в {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"В корне YAML должен быть словарь: {config_path}")
    return config


def resolve_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root) / path).resolve()


def select_device(value: str = "auto") -> torch.device:
    """Select CPU/CUDA and fail clearly when requested CUDA is unavailable."""
    normalized = value.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("В конфигурации выбрана CUDA, но GPU недоступен")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("training.device должен быть auto, cpu или cuda")
    return torch.device(normalized)


def environment_info() -> dict[str, Any]:
    """Return JSON-serializable software and GPU information."""
    packages = {}
    for package in (
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "albumentations",
        "opencv-python-headless",
        "pandas",
        "mlflow",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a UTF-8 JSON file, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return destination
