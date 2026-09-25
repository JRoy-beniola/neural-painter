"""Tests for reconstruction metrics."""

import numpy as np
import pytest

from painter.metrics import reconstruction_metrics


def test_identical_images_have_zero_mse_and_unit_ssim() -> None:
    image = np.full((16, 16, 3), 0.5, dtype=np.float32)

    metrics = reconstruction_metrics(image, image.copy())

    assert metrics["mse"] == 0.0
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["psnr"] == float("inf")


def test_shape_mismatch_is_rejected() -> None:
    left = np.zeros((16, 16, 3), dtype=np.float32)
    right = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        reconstruction_metrics(left, right)


def test_non_rgb_inputs_are_rejected() -> None:
    image = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        reconstruction_metrics(image, image)
