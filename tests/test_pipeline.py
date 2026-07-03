"""Fast pipeline tests built on a tiny synthetic Cityscapes-like dataset."""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as functional

from scripts.create_split import validate_manifest
from scripts.evaluate_clean import require_official_val_path, save_confusion_matrix
from scripts.evaluate_corruptions import (
    EXPECTED_CONDITION_COUNT,
    build_robustness_summary,
    validate_complete_results,
)
from src.corruptions import (
    CORRUPTION_MANIFEST_COLUMNS,
    CORRUPTION_NAMES,
    SEVERITY_LEVELS,
    CorruptionTransform,
    apply_corruption,
    corruption_seed,
    create_corruption_manifest,
    load_corruption_config,
)
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
from src.train import create_grad_scaler, train_model
from src.visualization import colorize_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRUPTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "corruptions.yaml"


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


def test_resume_completed_checkpoint_does_not_train_again(tmp_path: Path) -> None:
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
    best_path = checkpoint_dir / "unet_best.pt"
    last_path = checkpoint_dir / "unet_last.pt"
    torch.save(checkpoint, best_path)
    torch.save(checkpoint, last_path)
    history_path = tmp_path / "training_history_unet.csv"
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
    assert len(history) == 8, "Resume потерял уже записанные строки истории"
    assert returned_best == best_path
    assert returned_last == last_path


def test_cityscapes_preview_palette_handles_ignore_index() -> None:
    colored = colorize_mask(torch.tensor([[0, 18, 255]], dtype=torch.int64))
    assert colored.shape == (1, 3, 3)
    assert colored[0, 2].tolist() == [0, 0, 0], (
        "ignore_index=255 в preview должен отображаться чёрным"
    )


@pytest.fixture()
def corruption_input() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(12345)
    image = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    return image, load_corruption_config(CORRUPTION_CONFIG_PATH)


def test_corruptions_are_deterministic(corruption_input: tuple[np.ndarray, dict]) -> None:
    image, config = corruption_input
    original = image.copy()
    for corruption in CORRUPTION_NAMES:
        first = apply_corruption(image, "aachen_000001_000001", corruption, 2, config)
        second = apply_corruption(image, "aachen_000001_000001", corruption, 2, config)
        assert np.array_equal(first, second), (
            f"{corruption} недетерминирован для одинаковых image_id и severity"
        )
    assert corruption_seed("image_a", "gaussian_noise", 1) == corruption_seed(
        "image_a", "gaussian_noise", 1
    )
    assert corruption_seed("image_a", "gaussian_noise", 1) != corruption_seed(
        "image_b", "gaussian_noise", 1
    )
    assert np.array_equal(image, original), "Corruption изменил исходный RGB-массив in-place"


def test_corruption_config_matches_fixed_table() -> None:
    config = load_corruption_config(CORRUPTION_CONFIG_PATH)
    levels = {
        name: specification["levels"]
        for name, specification in config["corruptions"].items()
    }
    assert [levels["darkness"][severity]["factor"] for severity in SEVERITY_LEVELS] == [
        0.75,
        0.55,
        0.35,
    ]
    assert [levels["contrast"][severity]["factor"] for severity in SEVERITY_LEVELS] == [
        0.80,
        0.60,
        0.40,
    ]
    assert [
        (
            levels["gaussian_blur"][severity]["kernel_size"],
            levels["gaussian_blur"][severity]["sigma"],
        )
        for severity in SEVERITY_LEVELS
    ] == [(5, 1.0), (9, 2.0), (13, 3.0)]
    assert [
        levels["motion_blur"][severity]["kernel_size"]
        for severity in SEVERITY_LEVELS
    ] == [7, 15, 25]
    assert [
        levels["gaussian_noise"][severity]["sigma"]
        for severity in SEVERITY_LEVELS
    ] == [8.0, 16.0, 28.0]
    assert [
        levels["impulse_noise"][severity]["probability"]
        for severity in SEVERITY_LEVELS
    ] == [0.005, 0.015, 0.030]
    assert [levels["jpeg"][severity]["quality"] for severity in SEVERITY_LEVELS] == [
        70,
        40,
        15,
    ]
    assert [levels["fog"][severity]["alpha"] for severity in SEVERITY_LEVELS] == [
        0.15,
        0.30,
        0.45,
    ]


def test_corruptions_preserve_shape_dtype_and_range(
    corruption_input: tuple[np.ndarray, dict],
) -> None:
    image, config = corruption_input
    for corruption in CORRUPTION_NAMES:
        for severity in SEVERITY_LEVELS:
            result = apply_corruption(
                image, "bochum_000002_000001", corruption, severity, config
            )
            assert result.shape == image.shape, (
                f"{corruption} severity={severity} изменил размер изображения"
            )
            assert result.dtype == np.uint8, (
                f"{corruption} severity={severity} вернул {result.dtype}, ожидался uint8"
            )
            assert int(result.min()) >= 0 and int(result.max()) <= 255, (
                f"{corruption} severity={severity} вышел за диапазон 0..255"
            )


def test_corruption_does_not_change_mask(tiny_cityscapes: dict[str, Path]) -> None:
    config = load_corruption_config(CORRUPTION_CONFIG_PATH)
    clean_dataset = make_dataset(tiny_cityscapes)
    corrupted_dataset = CityscapesDataset(
        manifest_path=tiny_cityscapes["manifest"],
        dataset_root=tiny_cityscapes["root"],
        split="dev",
        train=False,
        width=384,
        height=192,
        image_corruption=CorruptionTransform("impulse_noise", 3, config),
    )
    for index in range(len(clean_dataset)):
        assert torch.equal(
            clean_dataset[index]["mask"], corrupted_dataset[index]["mask"]
        ), "Corruption изменил segmentation mask"


def test_all_corruption_severities_are_distinct(
    corruption_input: tuple[np.ndarray, dict],
) -> None:
    image, config = corruption_input
    for corruption in CORRUPTION_NAMES:
        outputs = [
            apply_corruption(
                image, "frankfurt_000003_000001", corruption, severity, config
            )
            for severity in SEVERITY_LEVELS
        ]
        assert not np.array_equal(outputs[0], outputs[1]), (
            f"{corruption}: severity 1 и 2 дали одинаковый результат"
        )
        assert not np.array_equal(outputs[1], outputs[2]), (
            f"{corruption}: severity 2 и 3 дали одинаковый результат"
        )


def test_corruption_manifest_contains_references_not_cached_images(
    tmp_path: Path,
) -> None:
    config = load_corruption_config(CORRUPTION_CONFIG_PATH)
    clean = pd.DataFrame(
        [
            {
                "image_id": "aachen_000001_000001",
                "image_path": "leftImg8bit/val/a.png",
                "mask_path": "gtFine/val/a.png",
                "split": "val",
            }
        ]
    )
    destination = create_corruption_manifest(
        clean, tmp_path / "corruption_manifest.csv", config, split="val"
    )
    manifest = pd.read_csv(destination)
    assert tuple(manifest.columns) == CORRUPTION_MANIFEST_COLUMNS
    assert len(manifest) == len(CORRUPTION_NAMES) * len(SEVERITY_LEVELS)
    assert manifest[["image_id", "corruption", "severity"]].duplicated().sum() == 0
    assert not any("corrupted" in column for column in manifest.columns), (
        "Manifest не должен ссылаться на сохранённые копии corrupted-изображений"
    )


def make_complete_corruption_results(
    model_values: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    family_by_corruption = {
        "darkness": "lighting",
        "contrast": "lighting",
        "gaussian_blur": "blur",
        "motion_blur": "blur",
        "gaussian_noise": "noise",
        "impulse_noise": "noise",
        "jpeg": "digital",
        "fog": "weather",
    }
    for model_name, values in model_values.items():
        clean_miou = values["clean"]
        rows.append(
            {
                "model": model_name,
                "corruption": "clean",
                "family": "clean",
                "severity": 0,
                "miou": clean_miou,
                "macro_dice": clean_miou,
                "delta_miou": 0.0,
                "retention": 1.0,
                "total_inference_seconds": 0.5,
                "mean_inference_ms_per_image": 1.0,
                "peak_gpu_memory_mb": values["memory"],
            }
        )
        for corruption in CORRUPTION_NAMES:
            for severity in SEVERITY_LEVELS:
                miou = values["corrupted"]
                rows.append(
                    {
                        "model": model_name,
                        "corruption": corruption,
                        "family": family_by_corruption[corruption],
                        "severity": severity,
                        "miou": miou,
                        "macro_dice": miou,
                        "delta_miou": clean_miou - miou,
                        "retention": miou / clean_miou,
                        "total_inference_seconds": 0.5,
                        "mean_inference_ms_per_image": 1.0,
                        "peak_gpu_memory_mb": values["memory"],
                    }
                )
    return pd.DataFrame(rows)


def test_robustness_summary_and_fixed_selection_order() -> None:
    results = make_complete_corruption_results(
        {
            "unet": {"clean": 0.70, "corrupted": 0.50, "memory": 900.0},
            "deeplabv3plus": {
                "clean": 0.90,
                "corrupted": 0.49,
                "memory": 800.0,
            },
            "pspnet": {"clean": 0.70, "corrupted": 0.50, "memory": 1000.0},
        }
    )
    validate_complete_results(results)
    assert len(results) == 3 * EXPECTED_CONDITION_COUNT
    summary = build_robustness_summary(
        results, ["lighting", "blur", "noise", "digital", "weather"]
    )
    # Primary robustness beats a larger clean score; the final tie uses memory.
    assert summary["model"].tolist() == ["unet", "pspnet", "deeplabv3plus"]
    assert summary.loc[0, "is_best_model"]
    assert summary.loc[0, "mean_corrupted_miou"] == pytest.approx(0.50)
    assert summary.loc[0, "family_noise_miou"] == pytest.approx(0.50)


def test_tracking_parameters_are_flattened() -> None:
    flattened = flatten_parameters(
        {"seed": 42, "training": {"epochs": 8}, "models": ["unet", "pspnet"]}
    )
    assert flattened == {
        "seed": 42,
        "training.epochs": 8,
        "models": "unet,pspnet",
    }


def test_clean_evaluation_rejects_train_path() -> None:
    require_official_val_path("leftImg8bit/val", "official_val_images")
    with pytest.raises(ValueError, match="официальный val"):
        require_official_val_path("leftImg8bit/train", "official_val_images")


def test_confusion_matrix_is_saved_with_class_labels(tmp_path: Path) -> None:
    confusion = torch.eye(19, dtype=torch.int64)
    destination = save_confusion_matrix(confusion, "unet", tmp_path)
    saved = pd.read_csv(destination)
    assert destination.name == "confusion_matrix_unet.csv"
    assert saved.shape == (19, 20), (
        "CSV confusion matrix должен содержать target_class и 19 prediction-столбцов"
    )
    assert saved["target_class"].tolist()[0] == "road"
