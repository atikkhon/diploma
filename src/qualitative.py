"""Persist standardized qualitative artifacts for diploma figure generators."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import IGNORE_INDEX
from src.metrics import CITYSCAPES_CLASS_NAMES
from src.visualization import CITYSCAPES_COLORS, PREDICTION_OVERLAY_ALPHA


QUALITATIVE_SCHEMA_VERSION = "cityscapes-qualitative/v1"
MANIFEST_COLUMNS = [
    "schema_version",
    "export_key",
    "run_name",
    "run_kind",
    "model_name",
    "encoder_name",
    "checkpoint_epoch",
    "dataset_split",
    "dataset_index",
    "image_id",
    "city",
    "sequence",
    "frame",
    "condition",
    "severity",
    "corruption_parameters_json",
    "image_width",
    "image_height",
    "input_path",
    "ground_truth_path",
    "prediction_path",
    "overlay_path",
    "metadata_path",
    "input_sha256",
    "ground_truth_sha256",
]


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_png(array: np.ndarray, path: Path) -> None:
    if array.dtype != np.uint8:
        raise ValueError(f"PNG должен быть uint8, получено {array.dtype}: {path.name}")
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"PNG должен быть HW или HWC, получено {array.shape}: {path.name}"
        )
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError(
            f"RGB PNG должен иметь 3 канала, получено {array.shape}: {path.name}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Не удалось сохранить PNG: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_class_schema(export_root: str | Path) -> Path:
    """Save the complete trainId/name/color mapping once per export."""
    root = Path(export_root).expanduser().resolve()
    destination = root / "class_schema.json"
    classes = [
        {
            "train_id": train_id,
            "name": class_name,
            "color": CITYSCAPES_COLORS[train_id].tolist(),
        }
        for train_id, class_name in enumerate(CITYSCAPES_CLASS_NAMES)
    ]
    _write_json(
        {
            "schema_version": QUALITATIVE_SCHEMA_VERSION,
            "num_classes": len(classes),
            "ignore_index": IGNORE_INDEX,
            "ignore_color": [0, 0, 0],
            "prediction_overlay_alpha": PREDICTION_OVERLAY_ALPHA,
            "classes": classes,
        },
        destination,
    )
    return destination


def save_qualitative_sample(
    export_root: str | Path,
    dataset_index: int,
    image_id: str,
    condition: str,
    severity: int | None,
    components: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save one clean/corrupted sample and return its manifest row."""
    root = Path(export_root).expanduser().resolve()
    if dataset_index < 0:
        raise ValueError("dataset_index должен быть >= 0")
    if not image_id or any(symbol in image_id for symbol in ("/", "\\", "..")):
        raise ValueError(f"Недопустимый image_id: {image_id}")
    if condition == "clean" and severity is not None:
        raise ValueError("Clean export не использует severity")
    if condition != "clean" and severity not in {1, 2, 3}:
        raise ValueError("Corruption export требует severity 1, 2 или 3")

    required = {"input", "ground_truth_trainid", "prediction_trainid", "overlay"}
    missing = required - set(components)
    if missing:
        raise ValueError(f"Не хватает компонентов: {sorted(missing)}")

    scene = root / f"index_{dataset_index:03d}__{image_id}"
    ground_truth_path = scene / "ground_truth_trainid.png"
    if condition == "clean":
        variant = scene / "clean"
    else:
        variant = scene / condition / f"severity_{severity}"
    input_path = variant / "input.png"
    prediction_path = variant / "prediction_trainid.png"
    overlay_path = variant / "overlay.png"
    metadata_path = variant / "metadata.json"

    _write_png(components["ground_truth_trainid"], ground_truth_path)
    _write_png(components["input"], input_path)
    _write_png(components["prediction_trainid"], prediction_path)
    _write_png(components["overlay"], overlay_path)

    input_hash = _sha256(input_path)
    ground_truth_hash = _sha256(ground_truth_path)
    file_paths = {
        "input": _relative(input_path, root),
        "ground_truth_trainid": _relative(ground_truth_path, root),
        "prediction_trainid": _relative(prediction_path, root),
        "overlay": _relative(overlay_path, root),
        "class_schema": "class_schema.json",
    }
    complete_metadata = dict(metadata)
    complete_metadata["schema_version"] = QUALITATIVE_SCHEMA_VERSION
    complete_metadata["paths_relative_to"] = "qualitative_export_root"
    complete_metadata["files"] = file_paths
    complete_metadata["sha256"] = {
        "input": input_hash,
        "ground_truth_trainid": ground_truth_hash,
    }
    _write_json(complete_metadata, metadata_path)

    dataset = complete_metadata["dataset"]
    run = complete_metadata["run"]
    model = complete_metadata["model"]
    checkpoint = complete_metadata["checkpoint"]
    corruption = complete_metadata["condition"]
    severity_value = 0 if severity is None else int(severity)
    return {
        "schema_version": QUALITATIVE_SCHEMA_VERSION,
        "export_key": f"{image_id}|{condition}|{severity_value}",
        "run_name": run["name"],
        "run_kind": run["kind"],
        "model_name": model["name"],
        "encoder_name": model["encoder_name"],
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset_split": dataset["split"],
        "dataset_index": dataset_index,
        "image_id": image_id,
        "city": dataset["city"],
        "sequence": dataset["sequence"],
        "frame": dataset["frame"],
        "condition": condition,
        "severity": severity_value,
        "corruption_parameters_json": json.dumps(
            corruption["parameters"], ensure_ascii=False, sort_keys=True
        ),
        "image_width": complete_metadata["image_width"],
        "image_height": complete_metadata["image_height"],
        "input_path": file_paths["input"],
        "ground_truth_path": file_paths["ground_truth_trainid"],
        "prediction_path": file_paths["prediction_trainid"],
        "overlay_path": file_paths["overlay"],
        "metadata_path": _relative(metadata_path, root),
        "input_sha256": input_hash,
        "ground_truth_sha256": ground_truth_hash,
    }


def upsert_manifest(rows: list[dict[str, Any]], manifest_path: str | Path) -> Path:
    """Replace matching export keys so repeated exports never create duplicates."""
    if not rows:
        raise ValueError("rows должен содержать хотя бы одну строку")
    destination = Path(manifest_path).expanduser().resolve()
    incoming = pd.DataFrame(rows)
    missing_columns = set(MANIFEST_COLUMNS) - set(incoming.columns)
    if missing_columns:
        raise ValueError(
            f"В новых строках manifest отсутствуют: {sorted(missing_columns)}"
        )
    incoming = incoming[MANIFEST_COLUMNS]
    if incoming["export_key"].duplicated().any():
        duplicate = incoming.loc[
            incoming["export_key"].duplicated(), "export_key"
        ].iloc[0]
        raise ValueError(f"Повторяющийся export_key в новых данных: {duplicate}")

    if destination.is_file():
        existing = pd.read_csv(
            destination,
            dtype={"sequence": str, "frame": str, "image_id": str},
        )
        missing_existing = set(MANIFEST_COLUMNS) - set(existing.columns)
        if missing_existing:
            raise ValueError(
                f"Существующий manifest имеет другую схему: {sorted(missing_existing)}"
            )
        existing = existing.loc[
            ~existing["export_key"].isin(incoming["export_key"]),
            MANIFEST_COLUMNS,
        ]
        result = pd.concat([existing, incoming], ignore_index=True)
    else:
        result = incoming

    result = result.sort_values(
        ["dataset_index", "condition", "severity"],
        kind="stable",
    ).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8")
    return destination
