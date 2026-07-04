"""The single supported image corruption: deterministic darkness."""

from typing import Callable

import numpy as np


DARKNESS_LEVELS = {1: 0.75, 2: 0.55, 3: 0.35}


def apply_darkness(image: np.ndarray, factor: float) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ожидалось RGB-изображение uint8 H×W×3")
    if not 0.0 < factor < 1.0:
        raise ValueError("Коэффициент darkness должен находиться между 0 и 1")
    return np.rint(image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def darkness_transform(factor: float) -> Callable[[np.ndarray, str], np.ndarray]:
    def transform(image: np.ndarray, image_id: str) -> np.ndarray:
        del image_id
        return apply_darkness(image, factor)

    return transform
