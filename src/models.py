"""Create the three baseline segmentation models used in the experiment."""

import segmentation_models_pytorch as smp
import torch.nn as nn


MODEL_ALIASES = {
    "unet": "unet",
    "u-net": "unet",
    "deeplabv3plus": "deeplabv3plus",
    "deeplabv3+": "deeplabv3plus",
    "pspnet": "pspnet",
}


def create_model(
    model_name: str,
    classes: int = 19,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Create U-Net, DeepLabV3+ or PSPNet with a shared encoder setup."""
    normalized_name = model_name.strip().lower().replace("_", "")
    canonical_name = MODEL_ALIASES.get(normalized_name)
    if canonical_name is None:
        supported = "unet, deeplabv3plus, pspnet"
        raise ValueError(
            f"Неизвестная модель '{model_name}'. Допустимые значения: {supported}"
        )
    if classes <= 0:
        raise ValueError("Число классов должно быть положительным")

    common_arguments = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": classes,
    }
    if canonical_name == "unet":
        return smp.Unet(**common_arguments)
    if canonical_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common_arguments)
    return smp.PSPNet(**common_arguments)
