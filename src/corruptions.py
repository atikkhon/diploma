"""Deterministic image corruptions used for manual robustness evaluation."""

import hashlib
from typing import Any, Callable

import cv2
import numpy as np


DARKNESS_LEVELS = {1: 0.75, 2: 0.55, 3: 0.35}
BRIGHTNESS_LEVELS = {1: 1.25, 2: 1.60, 3: 2.00}
GAUSSIAN_BLUR_LEVELS = {
    1: {"kernel_size": 3, "sigma": 0.7},
    2: {"kernel_size": 5, "sigma": 1.2},
    3: {"kernel_size": 9, "sigma": 2.0},
}
GAUSSIAN_NOISE_LEVELS = {1: 5.0, 2: 10.0, 3: 20.0}
JPEG_COMPRESSION_LEVELS = {1: 70, 2: 40, 3: 15}
FOG_LEVELS = {1: 0.15, 2: 0.30, 3: 0.45}

SUPPORTED_CORRUPTIONS = (
    "darkness",
    "brightness",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "fog",
)


def _validate_rgb_uint8(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ожидалось RGB-изображение uint8 H×W×3")


def _stable_seed(image_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(image).clip(0, 255).astype(np.uint8)


def apply_darkness(image: np.ndarray, factor: float) -> np.ndarray:
    _validate_rgb_uint8(image)
    if not 0.0 < factor < 1.0:
        raise ValueError("Коэффициент darkness должен находиться между 0 и 1")
    return _clip_uint8(image.astype(np.float32) * factor)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    _validate_rgb_uint8(image)
    if factor <= 1.0:
        raise ValueError("Коэффициент brightness должен быть больше 1")
    return _clip_uint8(image.astype(np.float32) * factor)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    _validate_rgb_uint8(image)
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size для gaussian_blur должен быть нечётным и > 1")
    if sigma <= 0.0:
        raise ValueError("sigma для gaussian_blur должен быть положительным")
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    image_id: str,
) -> np.ndarray:
    _validate_rgb_uint8(image)
    if sigma <= 0.0:
        raise ValueError("sigma для gaussian_noise должен быть положительным")
    generator = np.random.default_rng(_stable_seed(image_id, f"gaussian_noise_{sigma}"))
    noise = generator.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return _clip_uint8(image.astype(np.float32) + noise)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    _validate_rgb_uint8(image)
    if not 1 <= quality <= 100:
        raise ValueError("quality для jpeg_compression должен быть от 1 до 100")
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


def apply_fog(image: np.ndarray, alpha: float) -> np.ndarray:
    _validate_rgb_uint8(image)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha для fog должен находиться между 0 и 1")
    white = np.full_like(image, 255, dtype=np.float32)
    return _clip_uint8((1.0 - alpha) * image.astype(np.float32) + alpha * white)


def darkness_transform(factor: float) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_darkness(image, factor)

    return transform


def brightness_transform(factor: float) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_brightness(image, factor)

    return transform


def gaussian_blur_transform(
    kernel_size: int,
    sigma: float,
) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_gaussian_blur(image, kernel_size, sigma)

    return transform


def gaussian_noise_transform(sigma: float) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        return apply_gaussian_noise(image, sigma, image_id)

    return transform


def jpeg_compression_transform(quality: int) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_jpeg_compression(image, quality)

    return transform


def fog_transform(alpha: float) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_fog(image, alpha)

    return transform


def corruption_level(config: dict[str, Any], condition: str, severity: int) -> dict[str, Any]:
    if condition not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"Неизвестное искажение: {condition}")
    levels = config["corruptions"][condition]["levels"]
    level = levels.get(severity, levels.get(str(severity)))
    if not isinstance(level, dict):
        raise ValueError(f"Для {condition} выберите severity 1, 2 или 3")
    return level


def corruption_transform(
    condition: str,
    level: dict[str, Any],
) -> Callable[[np.ndarray, str], np.ndarray]:
    if condition == "darkness":
        return darkness_transform(float(level["factor"]))
    if condition == "brightness":
        return brightness_transform(float(level["factor"]))
    if condition == "gaussian_blur":
        return gaussian_blur_transform(
            int(level["kernel_size"]),
            float(level["sigma"]),
        )
    if condition == "gaussian_noise":
        return gaussian_noise_transform(float(level["sigma"]))
    if condition == "jpeg_compression":
        return jpeg_compression_transform(int(level["quality"]))
    if condition == "fog":
        return fog_transform(float(level["alpha"]))
    raise ValueError(f"Неизвестное искажение: {condition}")


def corruption_parameters(condition: str, level: dict[str, Any]) -> dict[str, float | int]:
    if condition == "darkness":
        return {"darkness_factor": float(level["factor"])}
    if condition == "brightness":
        return {"brightness_factor": float(level["factor"])}
    if condition == "gaussian_blur":
        return {
            "blur_kernel_size": int(level["kernel_size"]),
            "blur_sigma": float(level["sigma"]),
        }
    if condition == "gaussian_noise":
        return {"noise_sigma": float(level["sigma"])}
    if condition == "jpeg_compression":
        return {"jpeg_quality": int(level["quality"])}
    if condition == "fog":
        return {"fog_alpha": float(level["alpha"])}
    raise ValueError(f"Неизвестное искажение: {condition}")
