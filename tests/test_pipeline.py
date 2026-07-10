"""Fast tests for the independent segmentation pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as functional
import yaml

from scripts.create_split import validate_manifest
from scripts.evaluate_model import append_csv
from src.corruptions import DARKNESS_LEVELS, apply_darkness, darkness_transform
from src.dataset import (
    CityscapesDataset,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    cityscapes_manifest_dataset,
    discover_cityscapes_layout,
    find_cityscapes_pairs,
    prepare_train_id_masks,
    read_mask,
    validate_mask,
)
from src.experiment import load_run
from src.metrics import calculate_metrics, create_confusion_matrix, update_confusion_matrix
from src.models import MODEL_BUILDERS
from src.tracking import flatten_parameters
from src.train import create_grad_scaler, train_model
from src.visualization import colorize_mask, save_training_curves


@pytest.fixture()
def tiny_cityscapes(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "cityscapes"
    samples = [
        ("aachen", "000001", "000001", "train"),
        ("aachen", "000001", "000002", "train"),
        ("bochum", "000002", "000001", "dev"),
        ("bochum", "000002", "000002", "dev"),
    ]
    rows = []
    for number, (city, sequence, frame, split) in enumerate(samples):
        image_id = f"{city}_{sequence}_{frame}"
        image_dir = root / "leftImg8bit" / "train" / city
        mask_dir = root / "gtFine" / "train" / city
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        image[..., 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
        image[..., 1] = 40 + number * 20
        image[..., 2] = np.arange(24, dtype=np.uint8)[:, None] * 8
        mask = np.zeros((24, 48), dtype=np.uint8)
        mask[:, 12:24] = 1
        mask[:, 24:36] = 18
        mask[:, 36:] = 255
        image_path = image_dir / f"{image_id}{IMAGE_SUFFIX}"
        mask_path = mask_dir / f"{image_id}{MASK_SUFFIX}"
        assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert cv2.imwrite(str(mask_path), mask)
        rows.append(
            {
                "image_id": image_id,
                "image_path": image_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
                "city": city,
                "sequence": sequence,
                "split": split,
            }
        )
    manifest = tmp_path / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {"root": root, "manifest": manifest}


def make_dataset(
    paths: dict[str, Path],
    split: str = "dev",
    image_corruption=None,
) -> CityscapesDataset:
    return CityscapesDataset(
        manifest_path=paths["manifest"],
        dataset_root=paths["root"],
        split=split,
        train=False,
        width=384,
        height=192,
        image_corruption=image_corruption,
    )


def test_dataset_pairs_and_split(tiny_cityscapes: dict[str, Path]) -> None:
    pairs = find_cityscapes_pairs(
        tiny_cityscapes["root"], "leftImg8bit/train", "gtFine/train"
    )
    assert len(pairs) == 4
    frame = pd.read_csv(tiny_cityscapes["manifest"], dtype={"sequence": str})
    validate_manifest(frame)
    train_ids = set(frame.loc[frame["split"] == "train", "image_id"])
    dev_ids = set(frame.loc[frame["split"] == "dev", "image_id"])
    assert train_ids.isdisjoint(dev_ids)


def test_manifest_dataset_helper_writes_selected_split(
    tiny_cityscapes: dict[str, Path],
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "official_val_manifest.csv"
    dataset = cityscapes_manifest_dataset(
        dataset_root=tiny_cityscapes["root"],
        images_dir="leftImg8bit/train",
        masks_dir="gtFine/train",
        manifest_path=manifest,
        split="val",
        width=384,
        height=192,
        expected_count=4,
    )
    frame = pd.read_csv(manifest)
    assert len(dataset) == 4
    assert set(frame["split"]) == {"val"}
    assert frame["image_id"].iloc[0].startswith("aachen")


def test_dataset_shapes_dtypes_and_mask_resize(
    tiny_cityscapes: dict[str, Path],
) -> None:
    sample = make_dataset(tiny_cityscapes)[0]
    assert tuple(sample["image"].shape) == (3, 192, 384)
    assert tuple(sample["mask"].shape) == (192, 384)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 1, 18, 255}


def test_mask_validation_and_ignore_index() -> None:
    validate_mask(np.array([[0, 1, 18, 255]], dtype=np.uint8), "valid.png")
    with pytest.raises(ValueError, match="вне 0..18"):
        validate_mask(np.array([[19]], dtype=np.uint8), "invalid.png")
    target = torch.tensor([[[0, 255], [1, 18]]], dtype=torch.int64)
    logits = torch.zeros((1, 19, 2, 2), dtype=torch.float32)
    changed = logits.clone()
    changed[0, :, 0, 1] = torch.linspace(-50, 50, 19)
    assert torch.equal(
        functional.cross_entropy(logits, target, ignore_index=255, reduction="sum"),
        functional.cross_entropy(changed, target, ignore_index=255, reduction="sum"),
    )


def test_metrics_accumulate_across_batches() -> None:
    targets = torch.tensor([[0, 0, 1, 1, 255]], dtype=torch.int64)
    predictions = torch.tensor([[0, 1, 1, 1, 18]], dtype=torch.int64)
    complete = create_confusion_matrix(19)
    update_confusion_matrix(complete, predictions, targets, ignore_index=255)
    accumulated = create_confusion_matrix(19)
    update_confusion_matrix(accumulated, predictions[:, :2], targets[:, :2], 255)
    update_confusion_matrix(accumulated, predictions[:, 2:], targets[:, 2:], 255)
    assert torch.equal(complete, accumulated)
    metrics = calculate_metrics(complete)
    assert metrics["miou"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)
    assert metrics["pixel_accuracy"] == pytest.approx(0.75)


def test_nested_layout_and_label_conversion(tmp_path: Path) -> None:
    root = tmp_path / "download"
    image_root = root / "Cityscape Dataset" / "leftImg8bit"
    mask_root = root / "Fine Annotations" / "gtFine"
    for split in ("train", "val"):
        (image_root / split / "aachen").mkdir(parents=True)
        (mask_root / split / "aachen").mkdir(parents=True)
    layout = discover_cityscapes_layout(root)
    assert layout["train_images"] == image_root / "train"
    source = mask_root / "train" / "aachen" / "aachen_000001_000001_gtFine_labelIds.png"
    assert cv2.imwrite(str(source), np.array([[0, 7, 8, 33]], dtype=np.uint8))
    prepared = prepare_train_id_masks(mask_root / "train", tmp_path / "prepared")
    converted = read_mask(
        prepared / "aachen" / "aachen_000001_000001_gtFine_labelTrainIds.png"
    )
    assert converted.tolist() == [[255, 0, 1, 18]]


def test_expected_models_are_registered() -> None:
    assert set(MODEL_BUILDERS) == {"unet", "deeplabv3plus", "pspnet"}


def test_darkness_is_the_only_deterministic_corruption(
    tiny_cityscapes: dict[str, Path],
) -> None:
    assert DARKNESS_LEVELS == {1: 0.75, 2: 0.55, 3: 0.35}
    image = np.full((4, 5, 3), 200, dtype=np.uint8)
    outputs = [apply_darkness(image, factor) for factor in DARKNESS_LEVELS.values()]
    assert [int(output[0, 0, 0]) for output in outputs] == [150, 110, 70]
    assert all(output.dtype == np.uint8 and output.shape == image.shape for output in outputs)
    clean = make_dataset(tiny_cityscapes)[0]
    dark = make_dataset(
        tiny_cityscapes,
        image_corruption=darkness_transform(DARKNESS_LEVELS[2]),
    )[0]
    assert torch.equal(clean["mask"], dark["mask"])
    assert not torch.equal(clean["image"], dark["image"])


def test_completed_resume_keeps_history(tmp_path: Path) -> None:
    model = torch.nn.Conv2d(3, 19, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scaler = create_grad_scaler(torch.device("cpu"), enabled=False)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = {
        "epoch": 8,
        "model_name": "unet",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_miou": 0.25,
        "config": {},
    }
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    torch.save(checkpoint, best_path)
    torch.save(checkpoint, last_path)
    history_path = tmp_path / "training_history.csv"
    pd.DataFrame({"epoch": range(1, 9), "dev_miou": [0.25] * 8}).to_csv(
        history_path, index=False
    )
    history, returned_best, returned_last = train_model(
        model=model,
        model_name="unet",
        train_loader=[],
        dev_loader=[],
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(ignore_index=255),
        device=torch.device("cpu"),
        epochs=8,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        config={"training": {"log_interval": 1}},
        resume_path=last_path,
    )
    assert len(history) == 8
    assert returned_best == best_path
    assert returned_last == last_path


def test_run_paths_are_isolated(tmp_path: Path) -> None:
    template = {
        "data": {},
        "model": {"name": "unet"},
        "training": {},
        "evaluation": {},
        "tracking": {},
    }
    paths = []
    for name in ("run_a", "run_b"):
        config = {
            **template,
            "run": {
                "name": name,
                "output_dir": str(tmp_path / "runs" / name),
                "model_dir": str(tmp_path / "models" / name),
            },
        }
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        _, _, run_paths = load_run(config_path)
        paths.append(run_paths)
    assert paths[0].root != paths[1].root
    assert paths[0].history != paths[1].history
    assert paths[0].best_checkpoint != paths[1].best_checkpoint
    assert paths[0].root.parent.name == "runs"
    assert paths[0].checkpoints.parent.name == "models"
    assert not paths[0].best_checkpoint.is_relative_to(paths[0].root)


def test_repeated_csv_results_are_appended(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation_results.csv"
    append_csv([{"evaluation_id": "first", "miou": 0.1}], destination)
    append_csv([{"evaluation_id": "second", "miou": 0.2}], destination)
    result = pd.read_csv(destination)
    assert result["evaluation_id"].tolist() == ["first", "second"]


def test_tracking_parameters_are_flattened() -> None:
    assert flatten_parameters(
        {"seed": 42, "training": {"epochs": 8}, "models": ["unet"]}
    ) == {"seed": 42, "training.epochs": 8, "models": "unet"}


def test_preview_palette_handles_ignore_index() -> None:
    colored = colorize_mask(torch.tensor([[0, 18, 255]], dtype=torch.int64))
    assert colored.shape == (1, 3, 3)
    assert colored[0, 2].tolist() == [0, 0, 0]


def test_training_curves_are_saved(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [1.0, 0.8],
            "dev_loss": [1.1, 0.9],
            "dev_miou": [0.2, 0.3],
            **{f"dev_iou_{name}": [0.1, 0.2] for name in ("road", "sidewalk")},
        }
    )
    outputs = save_training_curves(history, tmp_path)
    assert [path.name for path in outputs] == [
        "training_loss_curve.png",
        "dev_miou_curve.png",
        "dev_per_class_iou_curve.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
