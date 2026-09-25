"""CIELAB palette extraction for Phase 0."""

from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans


def extract_palette(image_rgb: np.ndarray, n_colors: int, *, random_state: int = 0) -> np.ndarray:
    """Return RGB palette centers learned in CIELAB space.

    image_rgb must be float RGB in [0, 1] with shape (H, W, 3).
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if n_colors < 1:
        raise ValueError("n_colors must be positive")
    if not np.isfinite(image_rgb).all() or image_rgb.min() < 0.0 or image_rgb.max() > 1.0:
        raise ValueError("image_rgb values must be finite and lie in [0, 1]")

    lab = rgb2lab(image_rgb)
    pixels = lab.reshape(-1, 3)
    if n_colors > len(pixels):
        raise ValueError("n_colors cannot exceed the number of pixels")

    model = KMeans(n_clusters=n_colors, n_init="auto", random_state=random_state)
    centers_lab = model.fit(pixels).cluster_centers_
    centers_rgb = lab2rgb(centers_lab.reshape(1, n_colors, 3)).reshape(n_colors, 3)
    return np.clip(centers_rgb, 0.0, 1.0)
