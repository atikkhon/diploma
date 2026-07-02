"""Smoke tests for Cityscapes pairing, masks and split isolation."""

import numpy as np
import pandas as pd
import pytest

from scripts.create_split import validate_manifest
from src.dataset import city_and_sequence, image_id_from_name, validate_mask


def test_cityscapes_name_is_parsed() -> None:
    image_id = image_id_from_name("aachen_000000_000019_leftImg8bit.png")
    assert image_id == "aachen_000000_000019"
    assert city_and_sequence(image_id) == ("aachen", "000000")


def test_invalid_train_id_is_rejected() -> None:
    mask = np.array([[0, 18, 255], [1, 19, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="0..18"):
        validate_mask(mask, "bad_mask.png")


def test_train_and_dev_groups_do_not_overlap() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["city_000001_1", "city_000001_2"],
            "city": ["city", "city"],
            "sequence": ["000001", "000001"],
            "split": ["train", "dev"],
        }
    )
    with pytest.raises(ValueError, match="city/sequence"):
        validate_manifest(frame)
