"""Canvas background estimation for Phase 0 painters."""

from __future__ import annotations

import numpy as np


def estimate_border_background(image_rgb: np.ndarray) -> tuple[int, int, int]:
    """Estimate a robust canvas background from image-border pixels.

    The input is float RGB in [0, 1]. The returned color is 8-bit RGB so it can
    be passed directly to the raster renderer.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if image_rgb.shape[0] < 2 or image_rgb.shape[1] < 2:
        raise ValueError("image_rgb must be at least 2x2")
    if not np.isfinite(image_rgb).all() or image_rgb.min() < 0.0 or image_rgb.max() > 1.0:
        raise ValueError("image_rgb values must be finite and lie in [0, 1]")

    top = image_rgb[0, :, :]
    bottom = image_rgb[-1, :, :]
    left = image_rgb[1:-1, 0, :]
    right = image_rgb[1:-1, -1, :]
    border = np.concatenate((top, bottom, left, right), axis=0)

    median = np.median(border, axis=0)
    return tuple(round(float(channel) * 255.0) for channel in median)
