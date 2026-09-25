"""Stroke sampling utilities for Phase 0 painters."""

from __future__ import annotations

import math

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel_h, sobel_v

from painter.stroke import Stroke


def _nearest_color(pixel: np.ndarray, palette: np.ndarray) -> tuple[float, float, float]:
    index = int(np.argmin(np.sum((palette - pixel) ** 2, axis=1)))
    return tuple(float(x) for x in palette[index])


def gradient_field(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return horizontal gradient, vertical gradient, and magnitude."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")

    gray = rgb2gray(image_rgb)
    gx = sobel_h(gray)
    gy = sobel_v(gray)
    return gx, gy, np.hypot(gx, gy)


def gradient_magnitude(image_rgb: np.ndarray) -> np.ndarray:
    """Return Sobel gradient magnitude for float RGB input."""
    _, _, magnitude = gradient_field(image_rgb)
    return magnitude


def _sampling_weights(base: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if base.shape != shape:
        raise ValueError("weight_map must have shape (H, W)")
    if not np.isfinite(base).all() or np.any(base < 0.0):
        raise ValueError("weight_map must be finite and non-negative")

    weights = base.astype(float, copy=True)
    if float(weights.sum()) <= 0.0:
        weights.fill(1.0)
    weights = weights.ravel()
    weights /= weights.sum()
    return weights


def sample_gradient_strokes(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    count: int,
    *,
    seed: int = 0,
    weight_map: np.ndarray | None = None,
    min_length: float = 0.015,
    max_length: float = 0.08,
    width: float = 0.012,
    opacity: float = 0.85,
) -> list[Stroke]:
    """Sample strokes whose centerlines follow local image structure.

    When weight_map is supplied, it controls where stroke centers are sampled.
    Stroke orientation still follows the image gradient field.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette must have shape (K, 3)")
    if not 0.0 < min_length <= max_length <= 1.0:
        raise ValueError("stroke lengths must satisfy 0 < min_length <= max_length <= 1")
    if not 0.0 < width <= 1.0:
        raise ValueError("width must lie in (0, 1]")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must lie in [0, 1]")

    gx, gy, magnitude = gradient_field(image_rgb)

    if weight_map is None:
        weights_2d = magnitude.astype(float)
        weights_2d += max(float(weights_2d.mean()) * 0.15, 1e-8)
    else:
        weights_2d = weight_map

    weights = _sampling_weights(weights_2d, magnitude.shape)

    rng = np.random.default_rng(seed)
    h, w, _ = image_rgb.shape
    flat_indices = rng.choice(h * w, size=count, replace=True, p=weights)

    strokes: list[Stroke] = []
    max_strength = float(magnitude.max()) + 1e-8

    for flat in flat_indices:
        y, x = divmod(int(flat), w)
        cx = x / max(w - 1, 1)
        cy = y / max(h - 1, 1)

        theta = math.atan2(float(gy[y, x]), float(gx[y, x])) + math.pi / 2.0
        normalized_strength = float(magnitude[y, x]) / max_strength
        length = min_length + (max_length - min_length) * normalized_strength

        dx = math.cos(theta) * length / 2.0
        dy = math.sin(theta) * length / 2.0
        bend = rng.normal(0.0, length * 0.08)

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


def sample_fill_strokes(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    count: int,
    *,
    seed: int = 0,
    weight_map: np.ndarray,
    min_length: float = 0.025,
    max_length: float = 0.10,
    width: float = 0.025,
    opacity: float = 0.72,
) -> list[Stroke]:
    """Sample broad, low-curvature strokes for smooth interior regions."""
    if count < 1:
        raise ValueError("count must be positive")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette must have shape (K, 3)")
    if not 0.0 < min_length <= max_length <= 1.0:
        raise ValueError("stroke lengths must satisfy 0 < min_length <= max_length <= 1")
    if not 0.0 < width <= 1.0:
        raise ValueError("width must lie in (0, 1]")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must lie in [0, 1]")

    _, _, magnitude = gradient_field(image_rgb)
    weights = _sampling_weights(weight_map, magnitude.shape)

    rng = np.random.default_rng(seed)
    h, w, _ = image_rgb.shape
    flat_indices = rng.choice(h * w, size=count, replace=True, p=weights)

    max_strength = float(magnitude.max()) + 1e-8
    strokes: list[Stroke] = []

    for flat in flat_indices:
        y, x = divmod(int(flat), w)
        cx = x / max(w - 1, 1)
        cy = y / max(h - 1, 1)

        normalized_strength = float(magnitude[y, x]) / max_strength
        length = max_length - (max_length - min_length) * normalized_strength
        theta = float(rng.uniform(0.0, math.pi))

        dx = math.cos(theta) * length / 2.0
        dy = math.sin(theta) * length / 2.0
        p0 = (float(np.clip(cx - dx, 0.0, 1.0)), float(np.clip(cy - dy, 0.0, 1.0)))
        p2 = (float(np.clip(cx + dx, 0.0, 1.0)), float(np.clip(cy + dy, 0.0, 1.0)))
        p1 = (cx, cy)

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
