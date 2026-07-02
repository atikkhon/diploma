"""Load paired Cityscapes images and trainId masks with safe transforms."""

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelTrainIds.png"
NUM_CLASSES = 19
IGNORE_INDEX = 255
MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "mask_path",
    "city",
    "sequence",
    "split",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_id_from_name(file_name: str) -> str:
    """Remove the required Cityscapes image suffix and return its image id."""
    if not file_name.endswith(IMAGE_SUFFIX):
        raise ValueError(
            f"Ожидался файл *{IMAGE_SUFFIX}, получено: {file_name}"
        )
    return file_name[: -len(IMAGE_SUFFIX)]


def mask_id_from_name(file_name: str) -> str:
    """Remove the required Cityscapes mask suffix and return its image id."""
    if not file_name.endswith(MASK_SUFFIX):
        raise ValueError(f"Ожидался файл *{MASK_SUFFIX}, получено: {file_name}")
    return file_name[: -len(MASK_SUFFIX)]


def city_and_sequence(image_id: str) -> tuple[str, str]:
    """Read city and sequence from ``city_sequence_frame`` image id."""
    parts = image_id.split("_")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        raise ValueError(
            "Неверное имя Cityscapes. Ожидался формат "
            f"city_sequence_frame, получено: {image_id}"
        )
    return parts[0], parts[1]


def find_cityscapes_pairs(
    dataset_root: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
) -> list[dict[str, str]]:
    """Find image/mask pairs and fail if either side has no matching basename."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Корень Cityscapes не найден или не является каталогом: {root}"
        )

    image_root = root / images_dir
    mask_root = root / masks_dir
    if not image_root.is_dir():
        raise FileNotFoundError(f"Каталог изображений не найден: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Каталог масок не найден: {mask_root}")

    image_paths = sorted(image_root.rglob(f"*{IMAGE_SUFFIX}"))
    mask_paths = sorted(mask_root.rglob(f"*{MASK_SUFFIX}"))
    if not image_paths:
        raise FileNotFoundError(
            f"В {image_root} не найдены изображения *{IMAGE_SUFFIX}"
        )
    if not mask_paths:
        raise FileNotFoundError(f"В {mask_root} не найдены маски *{MASK_SUFFIX}")

    masks_by_id: dict[str, Path] = {}
    for mask_path in mask_paths:
        mask_id = mask_id_from_name(mask_path.name)
        if mask_id in masks_by_id:
            raise ValueError(f"Найдены две маски для image_id={mask_id}")
        masks_by_id[mask_id] = mask_path

    pairs: list[dict[str, str]] = []
    seen_image_ids: set[str] = set()
    for image_path in image_paths:
        image_id = image_id_from_name(image_path.name)
        if image_id in seen_image_ids:
            raise ValueError(f"Найдены два изображения с image_id={image_id}")
        seen_image_ids.add(image_id)

        mask_path = masks_by_id.get(image_id)
        if mask_path is None:
            expected_name = f"{image_id}{MASK_SUFFIX}"
            raise FileNotFoundError(
                f"Для изображения {image_path} не найдена маска {expected_name}"
            )
        if mask_id_from_name(mask_path.name) != image_id:
            raise ValueError(
                f"Базовые имена изображения и маски не совпадают: "
                f"{image_path.name}, {mask_path.name}"
            )

        city, sequence = city_and_sequence(image_id)
        pairs.append(
            {
                "image_id": image_id,
                "image_path": image_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
                "city": city,
                "sequence": sequence,
            }
        )

    orphan_mask_ids = set(masks_by_id) - seen_image_ids
    if orphan_mask_ids:
        example = sorted(orphan_mask_ids)[0]
        raise FileNotFoundError(
            f"Найдена маска без соответствующего изображения: {example}{MASK_SUFFIX}"
        )
    return pairs


def validate_mask(mask: np.ndarray, mask_path: str | Path = "<mask>") -> None:
    """Require one-channel trainId mask containing only 0..18 and 255."""
    if mask.ndim != 2:
        raise ValueError(
            f"Маска должна быть одноканальной, shape={mask.shape}: {mask_path}"
        )
    invalid = np.setdiff1d(
        np.unique(mask), np.array([*range(NUM_CLASSES), IGNORE_INDEX])
    )
    if invalid.size:
        shown = ", ".join(map(str, invalid[:10].tolist()))
        raise ValueError(
            f"Маска содержит значения вне 0..18 и ignore_index=255 "
            f"({shown}): {mask_path}. Используйте *_gtFine_labelTrainIds.png."
        )


def read_mask(mask_path: str | Path) -> np.ndarray:
    """Read a mask unchanged and report a clear error for a broken file."""
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл маски не найден: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"OpenCV не удалось прочитать маску: {path}")
    validate_mask(mask, path)
    return mask


def build_transform(
    train: bool,
    width: int = 384,
    height: int = 192,
    horizontal_flip_probability: float = 0.5,
) -> A.Compose:
    """Build deterministic evaluation or flip-enabled training transforms."""
    if width <= 0 or height <= 0:
        raise ValueError("width и height должны быть положительными")
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability должен быть между 0 и 1")
    transforms: list[Any] = []
    if train:
        transforms.append(A.HorizontalFlip(p=horizontal_flip_probability))
    transforms.extend(
        [
            A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


class CityscapesDataset(Dataset):
    """Read one split from a manifest and return image, mask and image_id."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str,
        train: bool = False,
        width: int = 384,
        height: int = 192,
        transform: A.Compose | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Корень Cityscapes не найден: {self.dataset_root}"
            )

        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Manifest не найден: {manifest}. Сначала запустите create_split.py."
            )
        frame = pd.read_csv(manifest)
        missing_columns = set(MANIFEST_COLUMNS) - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"В manifest отсутствуют столбцы: {sorted(missing_columns)}"
            )
        if frame["image_id"].duplicated().any():
            duplicate = frame.loc[frame["image_id"].duplicated(), "image_id"].iloc[0]
            raise ValueError(f"Повторяющийся image_id в manifest: {duplicate}")
        if split not in {"train", "dev", "val"}:
            raise ValueError("split должен быть train, dev или val")
        if train and split != "train":
            raise ValueError("Случайные train-преобразования разрешены только для train")

        self.rows = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"В manifest нет строк для split={split}")
        if transform is not None and not train:
            raise ValueError(
                "Для dev/val используется только фиксированное преобразование "
                "без случайных аугментаций"
            )
        self.transform = transform or build_transform(train, width, height)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image_path = self.dataset_root / str(row["image_path"])
        mask_path = self.dataset_root / str(row["mask_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Файл изображения не найден: {image_path}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV не удалось прочитать изображение: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = read_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Размеры изображения {image.shape[:2]} и маски {mask.shape[:2]} "
                f"не совпадают для image_id={row['image_id']}"
            )

        transformed = self.transform(image=image, mask=mask)
        output_mask = transformed["mask"]
        if not isinstance(output_mask, torch.Tensor):
            output_mask = torch.as_tensor(output_mask)
        return {
            "image": transformed["image"],
            "mask": output_mask.long(),
            "image_id": str(row["image_id"]),
        }
