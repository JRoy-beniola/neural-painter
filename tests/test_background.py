"""Tests for canvas background estimation."""

import numpy as np
import pytest

from painter.background import estimate_border_background


def test_border_background_uses_robust_median() -> None:
    image = np.zeros((6, 6, 3), dtype=np.float32)
    image[2:4, 2:4] = (1.0, 1.0, 1.0)

    assert estimate_border_background(image) == (0, 0, 0)


def test_border_background_handles_non_black_border() -> None:
    image = np.full((4, 4, 3), (0.2, 0.4, 0.6), dtype=np.float32)

    assert estimate_border_background(image) == (51, 102, 153)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((1, 4, 3), dtype=np.float32),
    ],
)
def test_invalid_background_inputs_are_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        estimate_border_background(image)
