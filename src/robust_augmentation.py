"""Provide the fixed moderate augmentation policy for robust retraining."""

from copy import deepcopy
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


DEFAULT_ROBUST_POLICY: dict[str, Any] = {
    "horizontal_flip_probability": 0.5,
    "brightness_contrast_probability": 0.3,
    "brightness_factor_range": [0.80, 1.20],
    "contrast_factor_range": [0.80, 1.20],
    "heavy_corruption_probability": 0.3,
    "gaussian_blur_kernel_sizes": [3, 5, 7],
    "gaussian_blur_sigma_range": [0.5, 1.5],
    "gaussian_noise_sigma_range": [4.0, 12.0],
    "jpeg_quality_range": [60, 90],
}

DEFAULT_SEEN_CORRUPTIONS = [
    "darkness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg",
]
DEFAULT_UNSEEN_CORRUPTIONS = ["motion_blur", "impulse_noise", "fog"]


def resolve_robust_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the policy from YAML or use the same fixed defaults for old runtime YAML."""
    robust = config.get("robust_training", {})
    configured = robust.get("augmentation") if isinstance(robust, Mapping) else None
    policy = deepcopy(DEFAULT_ROBUST_POLICY if configured is None else configured)
    validate_robust_policy(policy)
    return policy


def _range(policy: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = policy.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"robust_training.augmentation.{key} должен иметь два числа")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"Неверный диапазон {key}: {value}")
    return low, high


def validate_robust_policy(policy: Mapping[str, Any]) -> None:
    """Validate probabilities and moderate parameter ranges without tuning them."""
    required = set(DEFAULT_ROBUST_POLICY)
    if set(policy) != required:
        missing = required - set(policy)
        extra = set(policy) - required
        raise ValueError(
            f"Неверные поля robust augmentation; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for key in (
        "horizontal_flip_probability",
        "brightness_contrast_probability",
        "heavy_corruption_probability",
    ):
        value = float(policy[key])
        if not 0.0 <= value <= 0.5:
            raise ValueError(f"{key} должен быть умеренным и находиться в [0, 0.5]")
    brightness = _range(policy, "brightness_factor_range")
    contrast = _range(policy, "contrast_factor_range")
    blur_sigma = _range(policy, "gaussian_blur_sigma_range")
    noise_sigma = _range(policy, "gaussian_noise_sigma_range")
    jpeg_quality = _range(policy, "jpeg_quality_range")
    if brightness[0] < 0.5 or brightness[1] > 1.5:
        raise ValueError("brightness_factor_range выходит за умеренный диапазон")
    if contrast[0] < 0.5 or contrast[1] > 1.5:
        raise ValueError("contrast_factor_range выходит за умеренный диапазон")
    if blur_sigma[0] < 0.0 or noise_sigma[0] < 0.0:
        raise ValueError("Sigma не может быть отрицательной")
    if jpeg_quality[0] < 1 or jpeg_quality[1] > 100:
        raise ValueError("jpeg_quality_range должен находиться в [1, 100]")
    kernels = policy["gaussian_blur_kernel_sizes"]
    if not isinstance(kernels, (list, tuple)) or not kernels:
        raise ValueError("gaussian_blur_kernel_sizes должен быть непустым списком")
    if any(int(kernel) < 1 or int(kernel) % 2 == 0 for kernel in kernels):
        raise ValueError("Все Gaussian blur kernels должны быть положительными нечётными")


class RobustTrainingTransform:
    """Resize, flip and apply at most one heavy image corruption per call."""

    def __init__(self, width: int, height: int, policy: Mapping[str, Any]) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width и height должны быть положительными")
        validate_robust_policy(policy)
        self.width = width
        self.height = height
        self.policy = deepcopy(dict(policy))
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def _brightness_or_contrast(self, image: np.ndarray) -> np.ndarray:
        floating = image.astype(np.float32)
        if np.random.random() < 0.5:
            low, high = _range(self.policy, "brightness_factor_range")
            result = floating * float(np.random.uniform(low, high))
        else:
            low, high = _range(self.policy, "contrast_factor_range")
            factor = float(np.random.uniform(low, high))
            mean = floating.mean(axis=(0, 1), keepdims=True)
            result = mean + factor * (floating - mean)
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    def _one_heavy_corruption(self, image: np.ndarray) -> np.ndarray:
        choice = int(np.random.randint(0, 3))
        if choice == 0:
            kernels = [int(value) for value in self.policy["gaussian_blur_kernel_sizes"]]
            kernel = int(np.random.choice(kernels))
            low, high = _range(self.policy, "gaussian_blur_sigma_range")
            sigma = float(np.random.uniform(low, high))
            return cv2.GaussianBlur(
                image,
                (kernel, kernel),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
        if choice == 1:
            low, high = _range(self.policy, "gaussian_noise_sigma_range")
            sigma = float(np.random.uniform(low, high))
            noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
            return np.clip(
                np.rint(image.astype(np.float32) + noise), 0, 255
            ).astype(np.uint8)

        low, high = _range(self.policy, "jpeg_quality_range")
        quality = int(np.random.randint(int(low), int(high) + 1))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise RuntimeError("OpenCV не удалось выполнить robust JPEG augmentation")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV не удалось декодировать robust JPEG augmentation")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Robust transform ожидает RGB uint8 H×W×3")
        if mask.ndim != 2:
            raise ValueError("Robust transform ожидает двумерную segmentation mask")

        if np.random.random() < float(self.policy["horizontal_flip_probability"]):
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

        if np.random.random() < float(
            self.policy["brightness_contrast_probability"]
        ):
            image = self._brightness_or_contrast(image)
        if np.random.random() < float(self.policy["heavy_corruption_probability"]):
            image = self._one_heavy_corruption(image)

        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(normalized.transpose(2, 0, 1))
        ).to(torch.float32)
        return {"image": image_tensor, "mask": np.ascontiguousarray(mask)}
