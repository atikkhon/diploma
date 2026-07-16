"""Create and use a deterministic per-epoch training plan."""

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

from src.corruptions import (
    apply_brightness,
    apply_darkness,
    apply_gaussian_blur,
    apply_jpeg_compression,
)


ROBUST_CORRUPTIONS = (
    "darkness",
    "brightness",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
)
TRAINING_PLAN_COLUMNS = (
    "epoch",
    "position",
    "image_id",
    "corruption",
    "factor",
    "kernel_size",
    "sigma",
    "quality",
    "noise_seed",
)


def read_train_image_ids(split_manifest: str | Path) -> list[str]:
    """Return a stable sorted list of train image ids from split_manifest.csv."""
    path = Path(split_manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest не найден: {path}")
    frame = pd.read_csv(path, dtype={"image_id": str, "split": str})
    missing = {"image_id", "split"} - set(frame.columns)
    if missing:
        raise ValueError(f"В split manifest отсутствуют столбцы: {sorted(missing)}")
    train = frame.loc[frame["split"] == "train", "image_id"].astype(str)
    if train.empty:
        raise ValueError("Split manifest не содержит train-изображений")
    if train.duplicated().any():
        duplicate = train.loc[train.duplicated()].iloc[0]
        raise ValueError(f"Повторяющийся train image_id: {duplicate}")
    return sorted(train.tolist())


def _validate_robust_settings(augmentation: dict[str, Any] | None) -> list[str]:
    if augmentation is None:
        return []
    if str(augmentation.get("policy", "")).lower() != "robust":
        raise ValueError("Блок augmentation предназначен только для policy=robust")
    fraction = float(augmentation["corrupted_fraction"])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("augmentation.corrupted_fraction должен быть между 0 и 1")
    enabled = [
        name
        for name in ROBUST_CORRUPTIONS
        if augmentation.get(name, {}).get("enabled", False)
    ]
    if fraction > 0.0 and not enabled:
        raise ValueError("Robust-план требует хотя бы одно enabled-искажение")

    if "darkness" in enabled:
        settings = augmentation["darkness"]
        minimum = float(settings["min_factor"])
        maximum = float(settings["max_factor"])
        if not 0.0 < minimum <= maximum < 1.0:
            raise ValueError("darkness factor должен быть между 0 и 1")
    if "brightness" in enabled:
        settings = augmentation["brightness"]
        minimum = float(settings["min_factor"])
        maximum = float(settings["max_factor"])
        if not 1.0 < minimum <= maximum <= 3.0:
            raise ValueError("brightness factor должен быть больше 1 и не больше 3")
    if "gaussian_blur" in enabled:
        settings = augmentation["gaussian_blur"]
        kernels = [int(value) for value in settings["kernel_sizes"]]
        if not kernels or any(value <= 1 or value % 2 == 0 for value in kernels):
            raise ValueError("gaussian_blur kernel_sizes должны быть нечётными и > 1")
        if not 0.0 < float(settings["sigma_min"]) <= float(settings["sigma_max"]):
            raise ValueError("gaussian_blur sigma должен быть положительным")
    if "gaussian_noise" in enabled:
        settings = augmentation["gaussian_noise"]
        if not 0.0 < float(settings["sigma_min"]) <= float(settings["sigma_max"]):
            raise ValueError("gaussian_noise sigma должен быть положительным")
    if "jpeg_compression" in enabled:
        settings = augmentation["jpeg_compression"]
        minimum = int(settings["quality_min"])
        maximum = int(settings["quality_max"])
        if not 1 <= minimum <= maximum <= 100:
            raise ValueError("jpeg_compression quality должен быть от 1 до 100")
    return enabled


def _empty_plan_row(epoch: int, position: int, image_id: str) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "position": position,
        "image_id": image_id,
        "corruption": "clean",
        "factor": np.nan,
        "kernel_size": np.nan,
        "sigma": np.nan,
        "quality": np.nan,
        "noise_seed": np.nan,
    }


def _robust_assignments(
    image_ids: Sequence[str],
    epoch: int,
    rng: np.random.Generator,
    augmentation: dict[str, Any],
    enabled: Sequence[str],
) -> dict[str, str]:
    corrupted_count = math.floor(
        len(image_ids) * float(augmentation["corrupted_fraction"]) + 0.5
    )
    selected_indices = rng.permutation(len(image_ids))[:corrupted_count]
    offset = (epoch - 1) % len(enabled)
    corruptions = [enabled[(offset + index) % len(enabled)] for index in range(corrupted_count)]
    rng.shuffle(corruptions)
    return {
        image_ids[int(image_index)]: corruption
        for image_index, corruption in zip(selected_indices, corruptions, strict=True)
    }


def _validate_plan_parameters(plan: pd.DataFrame) -> None:
    parameter_columns = ("factor", "kernel_size", "sigma", "quality", "noise_seed")
    clean_rows = plan.loc[plan["corruption"] == "clean", parameter_columns]
    if clean_rows.notna().to_numpy().any():
        raise ValueError("Clean-строки training plan не должны содержать параметры искажения")

    factor = pd.to_numeric(plan["factor"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    kernel = pd.to_numeric(plan["kernel_size"], errors="coerce")
    sigma = pd.to_numeric(plan["sigma"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    quality = pd.to_numeric(plan["quality"], errors="coerce")
    noise_seed = pd.to_numeric(plan["noise_seed"], errors="coerce")
    valid = pd.Series(True, index=plan.index)

    darkness = plan["corruption"] == "darkness"
    valid.loc[darkness] = factor.loc[darkness].gt(0.0) & factor.loc[darkness].lt(1.0)
    brightness = plan["corruption"] == "brightness"
    valid.loc[brightness] = factor.loc[brightness].gt(1.0)
    blur = plan["corruption"] == "gaussian_blur"
    valid.loc[blur] = (
        kernel.loc[blur].gt(1.0)
        & kernel.loc[blur].mod(1).eq(0)
        & kernel.loc[blur].mod(2).eq(1)
        & sigma.loc[blur].gt(0.0)
    )
    noise = plan["corruption"] == "gaussian_noise"
    valid.loc[noise] = (
        sigma.loc[noise].gt(0.0)
        & noise_seed.loc[noise].ge(0.0)
        & noise_seed.loc[noise].lt(2**32)
        & noise_seed.loc[noise].mod(1).eq(0)
    )
    jpeg = plan["corruption"] == "jpeg_compression"
    valid.loc[jpeg] = (
        quality.loc[jpeg].between(1, 100) & quality.loc[jpeg].mod(1).eq(0)
    )

    if not valid.all():
        row = plan.loc[valid.index[~valid][0]]
        raise ValueError(
            "Training plan содержит неверные параметры: "
            f"epoch={row['epoch']}, image_id={row['image_id']}, "
            f"corruption={row['corruption']}"
        )


def _fill_corruption_parameters(
    row: dict[str, Any],
    corruption: str,
    augmentation: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    row["corruption"] = corruption
    settings = augmentation[corruption]
    if corruption in {"darkness", "brightness"}:
        row["factor"] = float(
            rng.uniform(float(settings["min_factor"]), float(settings["max_factor"]))
        )
    elif corruption == "gaussian_blur":
        row["kernel_size"] = int(rng.choice(settings["kernel_sizes"]))
        row["sigma"] = float(
            rng.uniform(float(settings["sigma_min"]), float(settings["sigma_max"]))
        )
    elif corruption == "gaussian_noise":
        row["sigma"] = float(
            rng.uniform(float(settings["sigma_min"]), float(settings["sigma_max"]))
        )
        row["noise_seed"] = int(rng.integers(0, 2**32, dtype=np.uint64))
    elif corruption == "jpeg_compression":
        row["quality"] = int(
            rng.integers(int(settings["quality_min"]), int(settings["quality_max"]) + 1)
        )
    else:
        raise ValueError(f"Неизвестное robust-искажение: {corruption}")


def validate_training_plan(
    plan: pd.DataFrame,
    train_image_ids: Sequence[str],
    epochs: int,
) -> None:
    """Validate complete coverage, unique positions and unique epoch orders."""
    missing_columns = set(TRAINING_PLAN_COLUMNS) - set(plan.columns)
    if missing_columns:
        raise ValueError(f"В training plan отсутствуют столбцы: {sorted(missing_columns)}")
    if epochs <= 0:
        raise ValueError("Число эпох training plan должно быть положительным")
    expected_ids = list(train_image_ids)
    expected_set = set(expected_ids)
    expected_epochs = set(range(1, epochs + 1))
    actual_epochs = set(plan["epoch"].astype(int).tolist())
    if actual_epochs != expected_epochs:
        raise ValueError(
            "Training plan содержит неверные эпохи: "
            f"ожидалось {sorted(expected_epochs)}, получено {sorted(actual_epochs)}"
        )
    if plan.duplicated(["epoch", "position"]).any():
        raise ValueError("Training plan содержит повторяющуюся пару epoch/position")
    if plan.duplicated(["epoch", "image_id"]).any():
        raise ValueError("Training plan содержит повторяющуюся пару epoch/image_id")

    used_orders: set[tuple[str, ...]] = set()
    for epoch in range(1, epochs + 1):
        epoch_rows = plan.loc[plan["epoch"].astype(int) == epoch].copy()
        epoch_rows["position"] = epoch_rows["position"].astype(int)
        epoch_rows = epoch_rows.sort_values("position")
        positions = epoch_rows["position"].tolist()
        if positions != list(range(len(expected_ids))):
            raise ValueError(f"Эпоха {epoch}: позиции должны идти от 0 до {len(expected_ids) - 1}")
        epoch_ids = epoch_rows["image_id"].astype(str).tolist()
        epoch_set = set(epoch_ids)
        if epoch_set != expected_set:
            missing = sorted(expected_set - epoch_set)
            extra = sorted(epoch_set - expected_set)
            raise ValueError(
                f"Эпоха {epoch} не соответствует train split. "
                f"Отсутствуют: {missing[:5]}; лишние: {extra[:5]}"
            )
        order = tuple(epoch_ids)
        if order in used_orders:
            raise ValueError(f"Эпоха {epoch} повторяет порядок предыдущей эпохи")
        used_orders.add(order)

    allowed = {"clean", *ROBUST_CORRUPTIONS}
    unknown = set(plan["corruption"].astype(str)) - allowed
    if unknown:
        raise ValueError(f"Training plan содержит неизвестные искажения: {sorted(unknown)}")
    _validate_plan_parameters(plan)


def generate_training_plan(
    train_image_ids: Sequence[str],
    epochs: int,
    seed: int,
    augmentation: dict[str, Any] | None = None,
    prefix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create missing epochs while preserving an optional validated prefix."""
    image_ids = sorted(str(image_id) for image_id in train_image_ids)
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Train image_id должны быть непустыми и уникальными")
    if epochs <= 0 or seed < 0:
        raise ValueError("epochs должен быть > 0, seed должен быть >= 0")
    if len(image_ids) < 10 and epochs > math.factorial(len(image_ids)):
        raise ValueError("Для такого числа изображений недостаточно уникальных перестановок")
    enabled = _validate_robust_settings(augmentation)

    if prefix is None or prefix.empty:
        result_rows: list[dict[str, Any]] = []
        start_epoch = 1
        used_orders: set[tuple[str, ...]] = set()
    else:
        prefix = prefix.loc[:, list(TRAINING_PLAN_COLUMNS)].copy()
        start_epoch = int(prefix["epoch"].max()) + 1
        validate_training_plan(prefix, image_ids, start_epoch - 1)
        result_rows = prefix.to_dict(orient="records")
        used_orders = {
            tuple(
                prefix.loc[prefix["epoch"].astype(int) == epoch]
                .sort_values("position")["image_id"]
                .astype(str)
                .tolist()
            )
            for epoch in range(1, start_epoch)
        }
    if start_epoch > epochs + 1:
        raise ValueError("Prefix training plan длиннее целевого числа эпох")

    for epoch in range(start_epoch, epochs + 1):
        attempt = 0
        while True:
            rng = np.random.default_rng([seed, epoch, attempt])
            permutation = rng.permutation(len(image_ids))
            ordered_ids = tuple(image_ids[int(index)] for index in permutation)
            if ordered_ids not in used_orders:
                used_orders.add(ordered_ids)
                break
            attempt += 1

        assignments: dict[str, str] = {}
        if augmentation is not None and enabled:
            assignments = _robust_assignments(
                image_ids,
                epoch,
                rng,
                augmentation,
                enabled,
            )
        for position, image_id in enumerate(ordered_ids):
            row = _empty_plan_row(epoch, position, image_id)
            corruption = assignments.get(image_id)
            if corruption is not None and augmentation is not None:
                _fill_corruption_parameters(row, corruption, augmentation, rng)
            result_rows.append(row)

    result = pd.DataFrame(result_rows, columns=TRAINING_PLAN_COLUMNS)
    validate_training_plan(result, image_ids, epochs)
    return result


def load_training_plan(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training plan не найден: {source}")
    return pd.read_csv(source, dtype={"image_id": str, "corruption": str})


def save_training_plan(plan: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(destination, index=False, encoding="utf-8")
    return destination


def prepare_training_plan(
    destination: str | Path,
    split_manifest: str | Path,
    epochs: int,
    seed: int,
    augmentation: dict[str, Any] | None = None,
    resume: bool = False,
    source_plan: str | Path | None = None,
    initial_epoch: int = 0,
) -> pd.DataFrame:
    """Load an existing plan or create a new/continued self-contained plan."""
    destination_path = Path(destination).expanduser().resolve()
    image_ids = read_train_image_ids(split_manifest)
    if resume:
        plan = load_training_plan(destination_path)
        validate_training_plan(plan, image_ids, epochs)
        return plan

    prefix = None
    if source_plan is not None:
        source = load_training_plan(source_plan)
        prefix = source.loc[source["epoch"].astype(int) <= initial_epoch].copy()
        validate_training_plan(prefix, image_ids, initial_epoch)
    plan = generate_training_plan(
        image_ids,
        epochs,
        seed,
        augmentation=augmentation,
        prefix=prefix,
    )
    save_training_plan(plan, destination_path)
    return plan


class TrainingPlanSampler(Sampler[tuple[int, int]]):
    """Yield ``(epoch, dataset_index)`` keys in the order stored in the plan."""

    def __init__(self, plan: pd.DataFrame, dataset_image_ids: Sequence[str]) -> None:
        index_by_id = {
            str(image_id): index for index, image_id in enumerate(dataset_image_ids)
        }
        if len(index_by_id) != len(dataset_image_ids):
            raise ValueError("Dataset содержит повторяющиеся image_id")
        self.indices_by_epoch: dict[int, list[int]] = {}
        for epoch, rows in plan.groupby(plan["epoch"].astype(int), sort=True):
            ordered = rows.sort_values("position")["image_id"].astype(str).tolist()
            self.indices_by_epoch[int(epoch)] = [index_by_id[image_id] for image_id in ordered]
        self.epoch = min(self.indices_by_epoch)

    def set_epoch(self, epoch: int) -> None:
        if epoch not in self.indices_by_epoch:
            raise ValueError(f"В training plan отсутствует эпоха {epoch}")
        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter((self.epoch, index) for index in self.indices_by_epoch[self.epoch])

    def __len__(self) -> int:
        return len(self.indices_by_epoch[self.epoch])


def apply_training_plan_corruption(
    image: np.ndarray,
    plan_row: Mapping[str, Any],
) -> np.ndarray:
    """Apply the exact RGB corruption recorded in one training-plan row."""
    corruption = str(plan_row["corruption"])
    if corruption == "clean":
        return image
    if corruption == "darkness":
        return apply_darkness(image, float(plan_row["factor"]))
    if corruption == "brightness":
        return apply_brightness(image, float(plan_row["factor"]))
    if corruption == "gaussian_blur":
        return apply_gaussian_blur(
            image,
            int(plan_row["kernel_size"]),
            float(plan_row["sigma"]),
        )
    if corruption == "gaussian_noise":
        sigma = float(plan_row["sigma"])
        generator = np.random.default_rng(int(plan_row["noise_seed"]))
        noise = generator.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return np.rint(image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    if corruption == "jpeg_compression":
        return apply_jpeg_compression(image, int(plan_row["quality"]))
    raise ValueError(f"Неизвестное искажение training plan: {corruption}")
