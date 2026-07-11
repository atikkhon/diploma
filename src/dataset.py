"""Load paired Cityscapes images and trainId masks with safe transforms."""

from pathlib import Path
from typing import Any, Callable

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
ROBUST_CORRUPTIONS = (
    "darkness",
    "brightness",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
)

# Official Cityscapes labelId -> 19-class trainId mapping.
LABEL_ID_TO_TRAIN_ID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
LABEL_ID_TO_TRAIN_ID[
    [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
] = np.arange(NUM_CLASSES, dtype=np.uint8)


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


def discover_cityscapes_layout(dataset_root: str | Path) -> dict[str, Path]:
    """Find nested leftImg8bit and gtFine directories under a downloaded root."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Корень загруженного датасета не найден: {root}")

    image_candidates = sorted(
        path for path in root.rglob("leftImg8bit")
        if (path / "train").is_dir() and (path / "val").is_dir()
    )
    mask_candidates = sorted(
        path for path in root.rglob("gtFine")
        if (path / "train").is_dir() and (path / "val").is_dir()
    )
    if len(image_candidates) != 1:
        raise FileNotFoundError(
            "Не удалось однозначно найти leftImg8bit с train/val внутри "
            f"{root}. Найдено вариантов: {len(image_candidates)}"
        )
    if len(mask_candidates) != 1:
        raise FileNotFoundError(
            "Не удалось однозначно найти gtFine с train/val внутри "
            f"{root}. Найдено вариантов: {len(mask_candidates)}"
        )
    return {
        "train_images": image_candidates[0] / "train",
        "val_images": image_candidates[0] / "val",
        "train_masks": mask_candidates[0] / "train",
        "val_masks": mask_candidates[0] / "val",
    }


def prepare_train_id_masks(
    source_split_dir: str | Path,
    output_split_dir: str | Path,
) -> Path:
    """Return existing trainId masks or convert labelIds into a writable cache."""
    source = Path(source_split_dir).expanduser().resolve()
    output = Path(output_split_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Каталог gtFine split не найден: {source}")

    train_id_masks = sorted(source.rglob(f"*{MASK_SUFFIX}"))
    label_id_masks = sorted(source.rglob("*_gtFine_labelIds.png"))
    if train_id_masks:
        if label_id_masks and len(train_id_masks) != len(label_id_masks):
            raise ValueError(
                f"В {source} найдено {len(train_id_masks)} labelTrainIds-масок и "
                f"{len(label_id_masks)} labelIds-масок; набор неполный"
            )
        print(f"Используются готовые labelTrainIds-маски: {source}")
        return source
    if not label_id_masks:
        raise FileNotFoundError(
            f"В {source} нет ни *{MASK_SUFFIX}, ни *_gtFine_labelIds.png"
        )

    output.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    reused_count = 0
    for source_path in label_id_masks:
        relative = source_path.relative_to(source)
        destination_name = source_path.name.replace(
            "_gtFine_labelIds.png", MASK_SUFFIX
        )
        destination = output / relative.parent / destination_name
        if destination.is_file() and destination.stat().st_size > 0:
            reused_count += 1
            continue

        label_ids = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if label_ids is None:
            raise ValueError(f"OpenCV не удалось прочитать labelIds-маску: {source_path}")
        if label_ids.ndim != 2 or label_ids.dtype != np.uint8:
            raise ValueError(
                f"Ожидалась одноканальная uint8 labelIds-маска, "
                f"получено shape={label_ids.shape}, dtype={label_ids.dtype}: {source_path}"
            )
        train_ids = LABEL_ID_TO_TRAIN_ID[label_ids]
        validate_mask(train_ids, source_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.stem + ".tmp.png")
        if not cv2.imwrite(str(temporary), train_ids):
            raise OSError(f"Не удалось сохранить trainId-маску: {temporary}")
        temporary.replace(destination)
        converted_count += 1

    prepared = sorted(output.rglob(f"*{MASK_SUFFIX}"))
    if len(prepared) != len(label_id_masks):
        raise ValueError(
            f"Подготовлено {len(prepared)} из {len(label_id_masks)} trainId-масок в {output}"
        )
    print(
        f"labelTrainIds готовы: {output} "
        f"(создано {converted_count}, использовано из кэша {reused_count})"
    )
    return output


def _path_for_manifest(path: Path, dataset_root: Path) -> str:
    """Keep portable relative paths when possible, otherwise keep an absolute path."""
    if path.is_relative_to(dataset_root):
        return path.relative_to(dataset_root).as_posix()
    return str(path)


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
                "image_path": _path_for_manifest(image_path, root),
                "mask_path": _path_for_manifest(mask_path, root),
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


def cityscapes_manifest_dataset(
    dataset_root: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    manifest_path: str | Path,
    split: str,
    width: int,
    height: int,
    image_corruption: Callable[[np.ndarray, str], np.ndarray] | None = None,
    expected_count: int | None = None,
) -> "CityscapesDataset":
    """Create a manifest from one Cityscapes split and return its dataset."""
    pairs = find_cityscapes_pairs(dataset_root, images_dir, masks_dir)
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(
            f"Cityscapes split {split} должен содержать {expected_count} пар, найдено {len(pairs)}"
        )
    frame = pd.DataFrame(pairs)
    frame["split"] = split
    destination = Path(manifest_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    return CityscapesDataset(
        manifest_path=destination,
        dataset_root=dataset_root,
        split=split,
        train=False,
        width=width,
        height=height,
        image_corruption=image_corruption,
    )


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


def read_mask(
    mask_path: str | Path,
    validate_values: bool = True,
) -> np.ndarray:
    """Read a mask, optionally checking all pixel values with ``np.unique``."""
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл маски не найден: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"OpenCV не удалось прочитать маску: {path}")
    if mask.ndim != 2:
        raise ValueError(
            f"Маска должна быть одноканальной, shape={mask.shape}: {path}"
        )
    if validate_values:
        validate_mask(mask, path)
    return mask


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(image).clip(0, 255).astype(np.uint8)


def _ensure_min_max(
    settings: dict[str, Any],
    min_key: str,
    max_key: str,
    name: str,
) -> tuple[float, float]:
    minimum = float(settings[min_key])
    maximum = float(settings[max_key])
    if minimum > maximum:
        raise ValueError(f"{name}: {min_key} должен быть <= {max_key}")
    return minimum, maximum


class RobustOneOf(A.ImageOnlyTransform):
    """Apply one randomly selected training corruption to the RGB image only."""

    def __init__(self, augmentation: dict[str, Any], p: float) -> None:
        super().__init__(p=p)
        self.augmentation = augmentation
        self.enabled_corruptions = [
            name
            for name in ROBUST_CORRUPTIONS
            if augmentation.get(name, {}).get("enabled", False)
        ]
        if not self.enabled_corruptions:
            raise ValueError("robust augmentation требует хотя бы одно enabled-искажение")
        self._validate_settings()

    def _validate_settings(self) -> None:
        if "darkness" in self.enabled_corruptions:
            minimum, maximum = _ensure_min_max(
                self.augmentation["darkness"],
                "min_factor",
                "max_factor",
                "darkness",
            )
            if not 0.0 < minimum <= maximum < 1.0:
                raise ValueError("darkness factor должен быть между 0 и 1")
        if "brightness" in self.enabled_corruptions:
            minimum, maximum = _ensure_min_max(
                self.augmentation["brightness"],
                "min_factor",
                "max_factor",
                "brightness",
            )
            if minimum <= 1.0:
                raise ValueError("brightness min_factor должен быть больше 1")
            if maximum > 3.0:
                raise ValueError("brightness max_factor слишком большой для обучения")
        if "gaussian_blur" in self.enabled_corruptions:
            settings = self.augmentation["gaussian_blur"]
            kernel_sizes = [int(value) for value in settings["kernel_sizes"]]
            if not kernel_sizes:
                raise ValueError("gaussian_blur.kernel_sizes не должен быть пустым")
            if any(value <= 1 or value % 2 == 0 for value in kernel_sizes):
                raise ValueError("gaussian_blur kernel_sizes должны быть нечётными и > 1")
            sigma_min, sigma_max = _ensure_min_max(
                settings,
                "sigma_min",
                "sigma_max",
                "gaussian_blur",
            )
            if sigma_min <= 0.0:
                raise ValueError("gaussian_blur sigma_min должен быть > 0")
        if "gaussian_noise" in self.enabled_corruptions:
            sigma_min, sigma_max = _ensure_min_max(
                self.augmentation["gaussian_noise"],
                "sigma_min",
                "sigma_max",
                "gaussian_noise",
            )
            if sigma_min <= 0.0 or sigma_max <= 0.0:
                raise ValueError("gaussian_noise sigma должен быть > 0")
        if "jpeg_compression" in self.enabled_corruptions:
            settings = self.augmentation["jpeg_compression"]
            quality_min = int(settings["quality_min"])
            quality_max = int(settings["quality_max"])
            if not 1 <= quality_min <= quality_max <= 100:
                raise ValueError("jpeg_compression quality должен быть от 1 до 100")

    def get_params(self) -> dict[str, Any]:
        corruption = str(np.random.choice(self.enabled_corruptions))
        if corruption == "darkness":
            minimum, maximum = _ensure_min_max(
                self.augmentation["darkness"],
                "min_factor",
                "max_factor",
                "darkness",
            )
            return {
                "corruption": corruption,
                "factor": float(np.random.uniform(minimum, maximum)),
            }
        if corruption == "brightness":
            minimum, maximum = _ensure_min_max(
                self.augmentation["brightness"],
                "min_factor",
                "max_factor",
                "brightness",
            )
            return {
                "corruption": corruption,
                "factor": float(np.random.uniform(minimum, maximum)),
            }
        if corruption == "gaussian_blur":
            settings = self.augmentation["gaussian_blur"]
            kernel_size = int(np.random.choice(settings["kernel_sizes"]))
            sigma = float(np.random.uniform(settings["sigma_min"], settings["sigma_max"]))
            return {
                "corruption": corruption,
                "kernel_size": kernel_size,
                "sigma": sigma,
            }
        if corruption == "gaussian_noise":
            settings = self.augmentation["gaussian_noise"]
            sigma = float(np.random.uniform(settings["sigma_min"], settings["sigma_max"]))
            return {"corruption": corruption, "sigma": sigma}
        settings = self.augmentation["jpeg_compression"]
        quality = int(np.random.randint(settings["quality_min"], settings["quality_max"] + 1))
        return {"corruption": corruption, "quality": quality}

    def apply(
        self,
        image: np.ndarray,
        corruption: str,
        factor: float = 1.0,
        kernel_size: int = 3,
        sigma: float = 1.0,
        quality: int = 85,
        **params: Any,
    ) -> np.ndarray:
        del params
        if corruption == "darkness":
            return _clip_uint8(image.astype(np.float32) * factor)
        if corruption == "brightness":
            return _clip_uint8(image.astype(np.float32) * factor)
        if corruption == "gaussian_blur":
            return cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                sigmaX=sigma,
                sigmaY=sigma,
            )
        if corruption == "gaussian_noise":
            noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
            return _clip_uint8(image.astype(np.float32) + noise)
        if corruption != "jpeg_compression":
            raise ValueError(f"Неизвестное robust-искажение: {corruption}")
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not success:
            raise ValueError("OpenCV не смог закодировать JPEG")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("OpenCV не смог декодировать JPEG")
        return decoded.astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


def build_transform(
    train: bool,
    width: int = 384,
    height: int = 192,
    horizontal_flip_probability: float = 0.5,
    augmentation_config: dict[str, Any] | None = None,
) -> A.Compose:
    """Build deterministic evaluation or baseline/robust training transforms."""
    if width <= 0 or height <= 0:
        raise ValueError("width и height должны быть положительными")
    augmentation = augmentation_config or {}
    policy = str(augmentation.get("policy", "baseline")).lower()
    if policy not in {"baseline", "robust"}:
        raise ValueError("augmentation.policy должен быть baseline или robust")
    horizontal_flip_probability = float(
        augmentation.get("horizontal_flip_probability", horizontal_flip_probability)
    )
    if not 0.0 <= horizontal_flip_probability <= 1.0:
        raise ValueError("horizontal_flip_probability должен быть между 0 и 1")
    robust_probability = float(augmentation.get("robust_one_of_probability", 0.0))
    if not 0.0 <= robust_probability <= 1.0:
        raise ValueError("robust_one_of_probability должен быть между 0 и 1")

    transforms: list[Any] = [
        A.Resize(
            height=height,
            width=width,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        )
    ]
    if train and horizontal_flip_probability > 0.0:
        transforms.append(A.HorizontalFlip(p=horizontal_flip_probability))
    if train and policy == "robust" and robust_probability > 0.0:
        transforms.append(RobustOneOf(augmentation=augmentation, p=robust_probability))
    transforms.extend(
        [
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
        image_corruption: Callable[[np.ndarray, str], np.ndarray] | None = None,
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
        if train and image_corruption is not None:
            raise ValueError(
                "Детерминированные corruption-преобразования предназначены только "
                "для validation/evaluation"
            )

        self.rows = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"В manifest нет строк для split={split}")
        if transform is not None and not train:
            raise ValueError(
                "Для dev/val используется только фиксированное преобразование "
                "без случайных аугментаций"
            )
        self.image_corruption = image_corruption
        self.transform = transform or build_transform(train, width, height)
        self.corruption_resize = None
        self.corruption_normalize = None
        if image_corruption is not None:
            # Corruption parameters are defined at the model input resolution.
            # The mask participates only in deterministic nearest-neighbour resize.
            self.corruption_resize = A.Resize(
                height=height,
                width=width,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            )
            self.corruption_normalize = A.Compose(
                [
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image_id = str(row["image_id"])
        image_path = self.dataset_root / str(row["image_path"])
        mask_path = self.dataset_root / str(row["mask_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Файл изображения не найден: {image_path}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV не удалось прочитать изображение: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # create_split.py already validates every full-resolution mask once.
        # Repeating np.unique over 2048x1024 pixels here would stall every epoch.
        mask = read_mask(mask_path, validate_values=False)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Размеры изображения {image.shape[:2]} и маски {mask.shape[:2]} "
                f"не совпадают для image_id={image_id}"
            )

        if self.image_corruption is not None:
            if self.corruption_resize is None or self.corruption_normalize is None:
                raise RuntimeError("Внутренние corruption-преобразования не созданы")
            resized = self.corruption_resize(image=image, mask=mask)
            corrupted_image = self.image_corruption(
                resized["image"].copy(), image_id
            )
            if not isinstance(corrupted_image, np.ndarray):
                raise TypeError(
                    f"Corruption для image_id={image_id} вернул не NumPy-массив"
                )
            if (
                corrupted_image.dtype != np.uint8
                or corrupted_image.ndim != 3
                or corrupted_image.shape[2] != 3
            ):
                raise ValueError(
                    "Corruption должен вернуть RGB uint8 H×W×3 до нормализации, "
                    f"получено shape={corrupted_image.shape}, "
                    f"dtype={corrupted_image.dtype}"
                )
            transformed = self.corruption_normalize(
                image=corrupted_image,
                mask=resized["mask"],
            )
        else:
            transformed = self.transform(image=image, mask=mask)
        output_mask = transformed["mask"]
        if not isinstance(output_mask, torch.Tensor):
            output_mask = torch.as_tensor(output_mask)
        return {
            "image": transformed["image"],
            "mask": output_mask.long(),
            "image_id": image_id,
        }
