"""Gradient-oriented stroke sampling for the Phase 0 baseline."""

from __future__ import annotations

import math

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel_h, sobel_v

from painter.stroke import Stroke


def _nearest_color(pixel: np.ndarray, palette: np.ndarray) -> tuple[float, float, float]:
    index = int(np.argmin(np.sum((palette - pixel) ** 2, axis=1)))
    return tuple(float(x) for x in palette[index])


def sample_gradient_strokes(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    count: int,
    *,
    seed: int = 0,
    min_length: float = 0.015,
    max_length: float = 0.08,
    width: float = 0.012,
    opacity: float = 0.85,
) -> list[Stroke]:
    """Sample strokes whose centerlines follow local image structure."""
    if count < 1:
        raise ValueError("count must be positive")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette must have shape (K, 3)")

    gray = rgb2gray(image_rgb)
    gx = sobel_h(gray)
    gy = sobel_v(gray)
    magnitude = np.hypot(gx, gy)

    weights = magnitude.ravel().astype(float)
    weights += max(float(weights.mean()) * 0.15, 1e-8)
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    h, w, _ = image_rgb.shape
    flat_indices = rng.choice(h * w, size=count, replace=True, p=weights)

    strokes: list[Stroke] = []
    for flat in flat_indices:
        y, x = divmod(int(flat), w)
        cx = x / max(w - 1, 1)
        cy = y / max(h - 1, 1)

        # Gradient points across local structure; the tangent follows it.
        theta = math.atan2(float(gy[y, x]), float(gx[y, x])) + math.pi / 2.0
        strength = float(magnitude[y, x])
        normalized_strength = strength / (float(magnitude.max()) + 1e-8)
        length = min_length + (max_length - min_length) * normalized_strength

        dx = math.cos(theta) * length / 2.0
        dy = math.sin(theta) * length / 2.0
        bend = rng.normal(0.0, length * 0.12)

        p0 = (float(np.clip(cx - dx, 0.0, 1.0)), float(np.clip(cy - dy, 0.0, 1.0)))
        p2 = (float(np.clip(cx + dx, 0.0, 1.0)), float(np.clip(cy + dy, 0.0, 1.0)))
        normal_x = -math.sin(theta)
        normal_y = math.cos(theta)
        p1 = (
            float(np.clip(cx + normal_x * bend, 0.0, 1.0)),
            float(np.clip(cy + normal_y * bend, 0.0, 1.0)),
        )

        strokes.append(
            Stroke(
                p0=p0,
                p1=p1,
                p2=p2,
                width=width,
                color=_nearest_color(image_rgb[y, x], palette),
                opacity=opacity,
            )
        )

    return strokes
