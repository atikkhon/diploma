"""Fast pipeline tests built on a tiny synthetic Cityscapes-like dataset."""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as functional
import yaml

from scripts.create_split import validate_manifest
from scripts.inspect_training_state import inspect_training_state
from src.dataset import (
    CityscapesDataset,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    discover_cityscapes_layout,
    find_cityscapes_pairs,
    prepare_train_id_masks,
    read_mask,
    validate_mask,
)
from src.metrics import (
    calculate_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.tracking import flatten_parameters
from src.train import create_grad_scaler


@pytest.fixture()
def tiny_cityscapes(tmp_path: Path) -> dict[str, Path]:
    """Create four valid PNG pairs in two independent city/sequence groups."""
    root = tmp_path / "cityscapes"
    samples = [
        ("aachen", "000001", "000001", "train"),
        ("aachen", "000001", "000002", "train"),
        ("bochum", "000002", "000001", "dev"),
        ("bochum", "000002", "000002", "dev"),
    ]
    rows: list[dict[str, str]] = []

    for number, (city, sequence, frame, split) in enumerate(samples):
        image_id = f"{city}_{sequence}_{frame}"
        image_dir = root / "leftImg8bit" / "train" / city
        mask_dir = root / "gtFine" / "train" / city
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # A small 2:1 RGB image with a gradient makes resize errors visible.
        image = np.zeros((24, 48, 3), dtype=np.uint8)
        image[..., 0] = np.arange(48, dtype=np.uint8)[None, :] * 5
        image[..., 1] = 40 + number * 20
        image[..., 2] = np.arange(24, dtype=np.uint8)[:, None] * 8

        # Sharp regions expose accidental bilinear interpolation of a mask.
        mask = np.zeros((24, 48), dtype=np.uint8)
        mask[:, 12:24] = 1
        mask[:, 24:36] = 18
        mask[:, 36:] = 255

        image_path = image_dir / f"{image_id}{IMAGE_SUFFIX}"
        mask_path = mask_dir / f"{image_id}{MASK_SUFFIX}"
        assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)), (
            f"Не удалось записать тестовое изображение: {image_path}"
        )
        assert cv2.imwrite(str(mask_path), mask), (
            f"Не удалось записать тестовую маску: {mask_path}"
        )
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


def make_dataset(paths: dict[str, Path], split: str = "dev") -> CityscapesDataset:
    return CityscapesDataset(
        manifest_path=paths["manifest"],
        dataset_root=paths["root"],
        split=split,
        train=False,
        width=384,
        height=192,
    )


def test_every_image_has_matching_mask(tiny_cityscapes: dict[str, Path]) -> None:
    pairs = find_cityscapes_pairs(
        tiny_cityscapes["root"], "leftImg8bit/train", "gtFine/train"
    )
    assert len(pairs) == 4, (
        "Ожидались четыре пары image-mask; проверьте суффиксы и структуру каталогов"
    )
    for pair in pairs:
        assert pair["image_id"] in Path(pair["image_path"]).name, (
            f"Изображение не соответствует image_id={pair['image_id']}"
        )
        assert pair["image_id"] in Path(pair["mask_path"]).name, (
            f"Маска не соответствует image_id={pair['image_id']}"
        )


def test_train_and_dev_do_not_overlap(tiny_cityscapes: dict[str, Path]) -> None:
    frame = pd.read_csv(tiny_cityscapes["manifest"], dtype={"sequence": str})
    validate_manifest(frame)
    train_ids = set(frame.loc[frame["split"] == "train", "image_id"])
    dev_ids = set(frame.loc[frame["split"] == "dev", "image_id"])
    assert train_ids.isdisjoint(dev_ids), (
        "Обнаружено пересечение image_id между train и dev"
    )


def test_mask_accepts_only_train_ids_and_ignore_index() -> None:
    valid = np.array([[0, 1, 18, 255]], dtype=np.uint8)
    validate_mask(valid, "valid_mask.png")

    invalid = np.array([[0, 18, 19, 254, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="вне 0..18"):
        validate_mask(invalid, "invalid_mask.png")


def test_dataset_shapes_and_dtypes(tiny_cityscapes: dict[str, Path]) -> None:
    sample = make_dataset(tiny_cityscapes)[0]
    assert tuple(sample["image"].shape) == (3, 192, 384), (
        f"Неверный размер изображения: {tuple(sample['image'].shape)}, "
        "ожидался (3, 192, 384)"
    )
    assert tuple(sample["mask"].shape) == (192, 384), (
        f"Неверный размер маски: {tuple(sample['mask'].shape)}, "
        "ожидался (192, 384)"
    )
    assert sample["image"].dtype == torch.float32, (
        f"Неверный dtype изображения: {sample['image'].dtype}, ожидался float32"
    )
    assert sample["mask"].dtype == torch.int64, (
        f"Неверный dtype маски: {sample['mask'].dtype}, ожидался int64"
    )


def test_nearest_resize_does_not_create_new_class_ids(
    tiny_cityscapes: dict[str, Path],
) -> None:
    resized_mask = make_dataset(tiny_cityscapes)[0]["mask"]
    values = set(torch.unique(resized_mask).tolist())
    expected = {0, 1, 18, 255}
    assert values == expected, (
        f"После resize появились промежуточные значения {sorted(values - expected)}; "
        "для маски должен использоваться только nearest neighbour"
    )


def test_validation_transform_is_deterministic(
    tiny_cityscapes: dict[str, Path],
) -> None:
    dataset = make_dataset(tiny_cityscapes)
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first["image"], second["image"]), (
        "Validation-преобразование изображения содержит случайность"
    )
    assert torch.equal(first["mask"], second["mask"]), (
        "Validation-преобразование маски содержит случайность"
    )


def test_ignore_index_is_preserved_and_ignored_by_loss(
    tiny_cityscapes: dict[str, Path],
) -> None:
    mask = make_dataset(tiny_cityscapes)[0]["mask"]
    assert torch.any(mask == 255), (
        "ignore_index=255 исчез после чтения или масштабирования маски"
    )

    target = torch.tensor([[[0, 255], [1, 18]]], dtype=torch.int64)
    logits = torch.zeros((1, 19, 2, 2), dtype=torch.float32)
    changed_at_ignored_pixel = logits.clone()
    changed_at_ignored_pixel[0, :, 0, 1] = torch.linspace(-50, 50, 19)
    first_loss = functional.cross_entropy(
        logits, target, ignore_index=255, reduction="sum"
    )
    second_loss = functional.cross_entropy(
        changed_at_ignored_pixel, target, ignore_index=255, reduction="sum"
    )
    assert torch.equal(first_loss, second_loss), (
        "Изменение logits в пикселе 255 повлияло на loss; "
        "проверьте ignore_index=255"
    )


def test_metrics_use_one_dataset_confusion_matrix() -> None:
    targets = torch.tensor([[0, 0, 1, 1, 255]], dtype=torch.int64)
    predictions = torch.tensor([[0, 1, 1, 1, 18]], dtype=torch.int64)

    complete = create_confusion_matrix(19)
    update_confusion_matrix(complete, predictions, targets, ignore_index=255)

    accumulated = create_confusion_matrix(19)
    update_confusion_matrix(
        accumulated, predictions[:, :2], targets[:, :2], ignore_index=255
    )
    update_confusion_matrix(
        accumulated, predictions[:, 2:], targets[:, 2:], ignore_index=255
    )
    assert torch.equal(complete, accumulated), (
        "Confusion matrix зависит от разбиения на batch; метрики должны "
        "накапливаться по всему набору"
    )

    metrics = calculate_metrics(complete)
    assert complete.sum().item() == 4, "Пиксель ignore_index=255 попал в матрицу"
    assert metrics["iou_per_class"][0] == pytest.approx(0.5)
    assert metrics["iou_per_class"][1] == pytest.approx(2.0 / 3.0)
    assert metrics["miou"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)
    assert metrics["macro_dice"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
    assert metrics["pixel_accuracy"] == pytest.approx(0.75)


def test_nested_kaggle_layout_and_label_id_conversion(tmp_path: Path) -> None:
    root = tmp_path / "kaggle_download"
    image_root = root / "Cityscape Dataset" / "leftImg8bit"
    gt_fine_root = root / "Fine Annotations" / "gtFine"
    for split in ("train", "val"):
        (image_root / split / "aachen").mkdir(parents=True)
        (gt_fine_root / split / "aachen").mkdir(parents=True)

    layout = discover_cityscapes_layout(root)
    assert layout["train_images"] == image_root / "train", (
        "Не найден вложенный KaggleHub-каталог leftImg8bit/train"
    )
    assert layout["train_masks"] == gt_fine_root / "train", (
        "Не найден вложенный KaggleHub-каталог gtFine/train"
    )

    label_ids = np.array([[0, 7, 8, 33]], dtype=np.uint8)
    source = gt_fine_root / "train" / "aachen"
    source_path = source / "aachen_000001_000001_gtFine_labelIds.png"
    assert cv2.imwrite(str(source_path), label_ids), (
        f"Не удалось создать тестовую labelIds-маску: {source_path}"
    )
    prepared_root = prepare_train_id_masks(
        gt_fine_root / "train", tmp_path / "prepared_gtFine" / "train"
    )
    prepared_path = (
        prepared_root / "aachen" / "aachen_000001_000001_gtFine_labelTrainIds.png"
    )
    converted = read_mask(prepared_path)
    assert converted.tolist() == [[255, 0, 1, 18]], (
        f"Неверное преобразование labelId → trainId: {converted.tolist()}"
    )


def test_current_grad_scaler_api_has_no_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = create_grad_scaler(torch.device("cpu"), enabled=False)
    assert not scaler.is_enabled()
    assert not any("deprecated" in str(item.message).lower() for item in caught), (
        "Создание GradScaler вызвало deprecation warning"
    )


def test_tracking_parameters_are_flattened() -> None:
    flattened = flatten_parameters(
        {"seed": 42, "training": {"epochs": 8}, "models": ["unet", "pspnet"]}
    )
    assert flattened == {
        "seed": 42,
        "training.epochs": 8,
        "models": "unet,pspnet",
    }


def test_inspector_recommends_skip_for_completed_model(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    checkpoint_dir = tmp_path / "checkpoints"
    history_dir = tmp_path / "outputs" / "metrics"
    config_dir.mkdir()
    checkpoint_dir.mkdir()
    history_dir.mkdir(parents=True)
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "training": {
                    "epochs": 8,
                    "checkpoint_dir": "checkpoints",
                    "history_dir": "outputs/metrics",
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {"epoch": 8, "model_name": "unet"}, checkpoint_dir / "unet_last.pt"
    )
    torch.save(
        {"epoch": 3, "model_name": "unet"}, checkpoint_dir / "unet_best.pt"
    )
    pd.DataFrame({"epoch": list(range(1, 9)), "dev_miou": [0.0] * 8}).to_csv(
        history_dir / "training_history_unet.csv", index=False
    )

    state = inspect_training_state(config_path, "unet")
    assert state["state"] == "completed"
    assert state["recommended_action"] == "skip"
