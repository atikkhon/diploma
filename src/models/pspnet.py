"""PSPNet model factory."""

from typing import Any

import segmentation_models_pytorch as smp
from torch import nn


def build_pspnet(classes: int, settings: dict[str, Any]) -> nn.Module:
    return smp.PSPNet(
        encoder_name=str(settings.get("encoder_name", "resnet34")),
        encoder_weights=settings.get("encoder_weights", "imagenet"),
        in_channels=3,
        classes=classes,
    )
