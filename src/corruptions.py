"""Apply deterministic Cityscapes corruptions to RGB images on-the-fly."""

import hashlib
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


CORRUPTION_NAMES = (
    "darkness",
    "contrast",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "impulse_noise",
    "jpeg",
    "fog",
)
SEVERITY_LEVELS = (1, 2, 3)
CORRUPTION_MANIFEST_COLUMNS = (
    "image_id",
    "image_path",
    "mask_path",
    "split",
    "corruption",
    "severity",
    "seed",
)


def corruption_seed(image_id: str, corruption: str, severity: int) -> int:
    """Derive a stable unsigned 64-bit seed from the required identifiers."""
    if not image_id:
        raise ValueError("image_id не должен быть пустым")
    payload = f"{image_id}|{corruption}|{severity}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _require_number(
    parameters: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float,
) -> float:
    value = parameters.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Параметр {key} должен быть числом, получено: {value}")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"Параметр {key} должен быть в диапазоне [{minimum}, {maximum}], "
            f"получено: {result}"
        )
    return result


def validate_corruption_config(config: Mapping[str, Any]) -> None:
    """Require exactly eight named corruptions with three valid levels."""
    if tuple(config.get("severity_levels", ())) != SEVERITY_LEVELS:
        raise ValueError("severity_levels должен быть равен [1, 2, 3]")
    corruptions = config.get("corruptions")
    if not isinstance(corruptions, Mapping):
        raise ValueError("В corruptions.yaml отсутствует словарь corruptions")
    if tuple(corruptions.keys()) != CORRUPTION_NAMES:
        raise ValueError(
            "corruptions должен содержать только восемь искажений в порядке: "
            + ", ".join(CORRUPTION_NAMES)
        )

    for name, specification in corruptions.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"Описание {name} должно быть словарём")
        levels = specification.get("levels")
        if not isinstance(levels, Mapping) or set(levels) != set(SEVERITY_LEVELS):
            raise ValueError(f"{name}.levels должен содержать уровни 1, 2 и 3")
        for severity in SEVERITY_LEVELS:
            parameters = levels[severity]
            if not isinstance(parameters, Mapping):
                raise ValueError(f"{name}.levels.{severity} должен быть словарём")
            if name in {"darkness", "contrast"}:
                _require_number(parameters, "factor", 0.0, 1.0)
            elif name == "gaussian_blur":
                kernel_size = int(
                    _require_number(parameters, "kernel_size", 1, 255)
                )
                if kernel_size % 2 == 0:
                    raise ValueError("kernel_size Gaussian blur должен быть нечётным")
                _require_number(parameters, "sigma", 0.0, 255.0)
            elif name == "motion_blur":
                kernel_size = int(
                    _require_number(parameters, "kernel_size", 1, 255)
                )
                if kernel_size % 2 == 0:
                    raise ValueError("kernel_size Motion blur должен быть нечётным")
            elif name == "gaussian_noise":
                _require_number(parameters, "sigma", 0.0, 255.0)
            elif name == "impulse_noise":
                _require_number(parameters, "probability", 0.0, 1.0)
            elif name == "jpeg":
                _require_number(parameters, "quality", 1, 100)
            elif name == "fog":
                _require_number(parameters, "alpha", 0.0, 1.0)


def load_corruption_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the fixed corruption YAML."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Конфигурация искажений не найдена: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Некорректный YAML в {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"В корне YAML должен быть словарь: {config_path}")
    validate_corruption_config(config)
    return config


def _validate_rgb_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image должен быть NumPy-массивом")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Ожидалось RGB-изображение H×W×3, получено {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"RGB до нормализации должен иметь dtype=uint8, получено {image.dtype}")


def _motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0
    rotation = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(
        kernel,
        rotation,
        (kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR,
    )
    total = float(kernel.sum())
    if total <= 0:
        raise RuntimeError("Не удалось построить kernel для motion blur")
    return kernel / total


def apply_corruption(
    image: np.ndarray,
    image_id: str,
    corruption: str,
    severity: int,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Return one deterministic corrupted RGB uint8 image without disk caching."""
    _validate_rgb_uint8(image)
    validate_corruption_config(config)
    if corruption not in CORRUPTION_NAMES:
        raise ValueError(
            f"Неизвестное искажение {corruption}; допустимы: "
            + ", ".join(CORRUPTION_NAMES)
        )
    if severity not in SEVERITY_LEVELS:
        raise ValueError("severity должен быть 1, 2 или 3")

    parameters = config["corruptions"][corruption]["levels"][severity]
    seed = corruption_seed(image_id, corruption, severity)
    rng = np.random.default_rng(seed)
    source = image.copy()
    floating = source.astype(np.float32)

    if corruption == "darkness":
        result = floating * float(parameters["factor"])
    elif corruption == "contrast":
        # Reduce contrast around each image's per-channel mean.
        mean = floating.mean(axis=(0, 1), keepdims=True)
        result = mean + float(parameters["factor"]) * (floating - mean)
    elif corruption == "gaussian_blur":
        kernel_size = int(parameters["kernel_size"])
        result = cv2.GaussianBlur(
            source,
            (kernel_size, kernel_size),
            sigmaX=float(parameters["sigma"]),
            sigmaY=float(parameters["sigma"]),
            borderType=cv2.BORDER_REFLECT_101,
        )
    elif corruption == "motion_blur":
        kernel_size = int(parameters["kernel_size"])
        angle = float(rng.uniform(0.0, 180.0))
        kernel = _motion_blur_kernel(kernel_size, angle)
        result = cv2.filter2D(
            source, -1, kernel, borderType=cv2.BORDER_REFLECT_101
        )
    elif corruption == "gaussian_noise":
        noise = rng.normal(
            loc=0.0,
            scale=float(parameters["sigma"]),
            size=source.shape,
        ).astype(np.float32)
        result = floating + noise
    elif corruption == "impulse_noise":
        probability = float(parameters["probability"])
        affected = rng.random(source.shape[:2]) < probability
        salt = rng.random(source.shape[:2]) < 0.5
        result = source.copy()
        result[affected & ~salt] = 0
        result[affected & salt] = 255
    elif corruption == "jpeg":
        bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(parameters["quality"])]
        )
        if not success:
            raise RuntimeError("OpenCV не удалось выполнить JPEG-сжатие")
        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError("OpenCV не удалось декодировать JPEG")
        result = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    else:  # fog
        alpha = float(parameters["alpha"])
        result = (1.0 - alpha) * floating + alpha * 255.0

    output = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    if output.shape != image.shape:
        raise RuntimeError(
            f"Искажение изменило размер {image.shape} -> {output.shape}"
        )
    return np.ascontiguousarray(output)


class CorruptionTransform:
    """Callable image-only transform suitable for ``CityscapesDataset``."""

    def __init__(
        self,
        corruption: str,
        severity: int,
        config: Mapping[str, Any],
    ) -> None:
        validate_corruption_config(config)
        if corruption not in CORRUPTION_NAMES:
            raise ValueError(f"Неизвестное искажение: {corruption}")
        if severity not in SEVERITY_LEVELS:
            raise ValueError("severity должен быть 1, 2 или 3")
        self.corruption = corruption
        self.severity = severity
        self.config = config

    def __call__(self, image: np.ndarray, image_id: str) -> np.ndarray:
        return apply_corruption(
            image,
            image_id,
            self.corruption,
            self.severity,
            self.config,
        )


def create_corruption_manifest(
    clean_manifest: str | Path | pd.DataFrame,
    output_path: str | Path,
    config: Mapping[str, Any],
    split: str | None = None,
) -> Path:
    """Expand clean rows into image/corruption/severity references, not images."""
    validate_corruption_config(config)
    if isinstance(clean_manifest, pd.DataFrame):
        frame = clean_manifest.copy()
    else:
        manifest_path = Path(clean_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Clean manifest не найден: {manifest_path}")
        frame = pd.read_csv(manifest_path)
    required = {"image_id", "image_path", "mask_path", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В clean manifest отсутствуют столбцы: {sorted(missing)}")
    if split is not None:
        frame = frame.loc[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"В clean manifest нет строк для split={split}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Clean manifest содержит повторяющиеся image_id")

    rows: list[dict[str, Any]] = []
    for clean_row in frame.sort_values("image_id").to_dict(orient="records"):
        for corruption in CORRUPTION_NAMES:
            for severity in SEVERITY_LEVELS:
                rows.append(
                    {
                        "image_id": str(clean_row["image_id"]),
                        "image_path": str(clean_row["image_path"]),
                        "mask_path": str(clean_row["mask_path"]),
                        "split": str(clean_row["split"]),
                        "corruption": corruption,
                        "severity": severity,
                        "seed": corruption_seed(
                            str(clean_row["image_id"]), corruption, severity
                        ),
                    }
                )
    result = pd.DataFrame(rows, columns=CORRUPTION_MANIFEST_COLUMNS)
    if result.duplicated(["image_id", "corruption", "severity"]).any():
        raise RuntimeError("Corruption manifest содержит повторяющиеся комбинации")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, encoding="utf-8")
    return destination


def save_corruption_examples(
    image: np.ndarray,
    image_id: str,
    output_path: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Save an 8×4 grid: clean plus severities 1, 2 and 3 per corruption."""
    _validate_rgb_uint8(image)
    validate_corruption_config(config)
    figure, axes = plt.subplots(
        len(CORRUPTION_NAMES),
        4,
        figsize=(14, 22),
        constrained_layout=True,
    )
    titles = ("Clean", "Severity 1", "Severity 2", "Severity 3")
    for row_index, corruption in enumerate(CORRUPTION_NAMES):
        panels = [image] + [
            apply_corruption(image, image_id, corruption, severity, config)
            for severity in SEVERITY_LEVELS
        ]
        for column_index, (panel, title) in enumerate(zip(panels, titles)):
            axis = axes[row_index, column_index]
            axis.imshow(panel)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title)
        axes[row_index, 0].set_ylabel(
            corruption, rotation=0, ha="right", va="center", fontsize=9
        )
    figure.suptitle(f"Corruption examples: {image_id}", fontsize=12)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Не удалось сохранить corruption grid: {destination}")
    return destination
