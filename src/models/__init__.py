"""Model registry. Add future architectures here without changing training code."""

from typing import Any, Callable

from torch import nn

from src.models.unet import build_unet


ModelBuilder = Callable[[int, dict[str, Any]], nn.Module]
MODEL_BUILDERS: dict[str, ModelBuilder] = {"unet": build_unet}


def create_model(
    model_name: str,
    classes: int,
    settings: dict[str, Any],
) -> nn.Module:
    name = model_name.strip().lower()
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"Неизвестная модель '{model_name}'. Доступно: {', '.join(MODEL_BUILDERS)}"
        )
    if classes <= 0:
        raise ValueError("Число классов должно быть положительным")
    return MODEL_BUILDERS[name](classes, settings)
