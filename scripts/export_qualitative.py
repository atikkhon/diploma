"""Export selected validation scenes as reusable qualitative artifacts."""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import SUPPORTED_CORRUPTIONS  # noqa: E402
from src.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.inference import (  # noqa: E402
    InferenceRun,
    build_official_val_dataset,
    load_inference_run,
    predict_masks,
)
from src.qualitative import (  # noqa: E402
    save_qualitative_sample,
    upsert_manifest,
    write_class_schema,
)
from src.visualization import (  # noqa: E402
    PREDICTION_OVERLAY_ALPHA,
    segmentation_components,
)


def _unique_values(values: list[Any], name: str) -> list[Any]:
    if not values:
        raise ValueError(f"{name} должен содержать хотя бы одно значение")
    result = list(dict.fromkeys(values))
    if len(result) != len(values):
        raise ValueError(f"{name} не должен содержать повторяющиеся значения")
    return result


def _metadata(
    run: InferenceRun,
    dataset,
    dataset_index: int,
    condition: str,
    severity: int | None,
    level: dict[str, Any],
) -> dict[str, Any]:
    image_id = str(dataset.rows.iloc[dataset_index]["image_id"])
    parts = image_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Неверный Cityscapes image_id: {image_id}")
    city, sequence, frame = parts
    row = dataset.rows.iloc[dataset_index]
    config = run.config
    run_config = config["run"]
    model_config = config["model"]
    return {
        "run": {
            "name": str(run_config["name"]),
            "kind": str(run_config.get("kind", "baseline")),
            "source_baseline_run": run_config.get("source_baseline_run"),
            "seed": int(config.get("seed", 42)),
            "training_epochs": int(config["training"]["epochs"]),
            "augmentation_policy": str(
                config.get("augmentation", {}).get("policy", "baseline")
            ),
            "config_from_export_root": "../../run_config.yaml",
        },
        "model": {
            "name": str(model_config["name"]),
            "encoder_name": str(model_config.get("encoder_name", "")),
            "encoder_weights": model_config.get("encoder_weights"),
        },
        "checkpoint": {
            "name": "best.pt",
            "epoch": run.checkpoint_epoch,
        },
        "dataset": {
            "name": "Cityscapes",
            "split": "official_val",
            "index": dataset_index,
            "image_id": image_id,
            "city": city,
            "sequence": sequence,
            "frame": frame,
            "image_path": str(row["image_path"]),
            "mask_path": str(row["mask_path"]),
        },
        "condition": {
            "name": condition,
            "severity": severity,
            "parameters": dict(level),
        },
        "image_width": int(config["data"]["image_width"]),
        "image_height": int(config["data"]["image_height"]),
        "num_classes": int(config["data"]["num_classes"]),
        "ignore_index": int(config["data"]["ignore_index"]),
        "normalization": {
            "name": "ImageNet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "prediction_overlay_alpha": PREDICTION_OVERLAY_ALPHA,
    }


def export_qualitative(
    config_path: str | Path,
    indices: list[int],
    conditions: list[str],
    severities: list[int],
) -> Path:
    """Export selected clean/corrupted scenes and upsert their manifest rows."""
    selected_indices = [int(value) for value in _unique_values(indices, "indices")]
    selected_conditions = [
        str(value) for value in _unique_values(conditions, "conditions")
    ]
    selected_severities = [
        int(value) for value in _unique_values(severities, "severities")
    ]
    allowed_conditions = {"clean", *SUPPORTED_CORRUPTIONS}
    unknown = set(selected_conditions) - allowed_conditions
    if unknown:
        raise ValueError(f"Неизвестные conditions: {sorted(unknown)}")
    if any(value not in {1, 2, 3} for value in selected_severities):
        raise ValueError("severities может содержать только 1, 2 и 3")

    run = load_inference_run(config_path)
    export_root = run.paths.predictions / "qualitative"
    write_class_schema(export_root)
    manifest_path = export_root / "manifest.csv"
    batch_size = int(run.config["evaluation"].get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("evaluation.batch_size должен быть > 0")

    variants: list[tuple[str, int | None]] = []
    for condition in selected_conditions:
        if condition == "clean":
            variants.append((condition, None))
        else:
            variants.extend((condition, severity) for severity in selected_severities)

    exported = 0
    for condition, severity in variants:
        dataset, level = build_official_val_dataset(run, condition, severity)
        for index in selected_indices:
            if index < 0 or index >= len(dataset):
                raise IndexError(f"Индекс должен быть от 0 до {len(dataset) - 1}: {index}")

        variant_rows: list[dict[str, Any]] = []
        for start in range(0, len(selected_indices), batch_size):
            batch_indices = selected_indices[start : start + batch_size]
            samples = [dataset[index] for index in batch_indices]
            images = torch.stack([sample["image"] for sample in samples])
            predictions = predict_masks(run, images)
            for index, sample, prediction in zip(batch_indices, samples, predictions):
                image_id = str(sample["image_id"])
                components = segmentation_components(
                    sample["image"],
                    sample["mask"],
                    prediction,
                )
                row = save_qualitative_sample(
                    export_root=export_root,
                    dataset_index=index,
                    image_id=image_id,
                    condition=condition,
                    severity=severity,
                    components=components,
                    metadata=_metadata(
                        run,
                        dataset,
                        index,
                        condition,
                        severity,
                        level,
                    ),
                )
                variant_rows.append(row)
                exported += 1
                print(
                    f"Exported: index={index}, image_id={image_id}, "
                    f"condition={condition}, severity={severity}",
                    flush=True,
                )
        upsert_manifest(variant_rows, manifest_path)

    print(f"Qualitative export: {export_root}")
    print(f"Exported variants: {exported}")
    print(f"Manifest: {manifest_path}")
    return export_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["clean", *SUPPORTED_CORRUPTIONS],
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_qualitative(
        args.config,
        args.indices,
        args.conditions,
        args.severities,
    )


if __name__ == "__main__":
    main()
