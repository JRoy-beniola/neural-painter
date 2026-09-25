"""Residual-driven coarse-to-fine painting for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from painter.background import estimate_border_background
from painter.renderer import render_strokes
from painter.sampling import gradient_magnitude, sample_gradient_strokes
from painter.stroke import Stroke


@dataclass(frozen=True, slots=True)
class PassConfig:
    """Configuration for one coarse-to-fine painting pass."""

    fraction: float
    min_length: float
    max_length: float
    width: float
    opacity: float


DEFAULT_PASSES = (
    PassConfig(0.15, 0.040, 0.140, 0.030, 0.90),
    PassConfig(0.25, 0.020, 0.090, 0.018, 0.88),
    PassConfig(0.60, 0.008, 0.050, 0.009, 0.85),
)


def _allocate_counts(total_strokes: int, passes: tuple[PassConfig, ...]) -> list[int]:
    if total_strokes < 1:
        raise ValueError("total_strokes must be positive")
    if not passes:
        raise ValueError("passes cannot be empty")

    raw = np.asarray([config.fraction for config in passes], dtype=float)
    if not np.isfinite(raw).all() or np.any(raw < 0.0) or float(raw.sum()) <= 0.0:
        raise ValueError("pass fractions must be finite, non-negative, and sum to > 0")

    normalized = raw / raw.sum()
    counts = np.floor(normalized * total_strokes).astype(int)
    remainder = total_strokes - int(counts.sum())

    if remainder:
        fractional = normalized * total_strokes - counts
        order = np.argsort(-fractional)
        for index in order[:remainder]:
            counts[int(index)] += 1

    return [int(value) for value in counts]


def _normalized_map(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max())
    if maximum <= 0.0:
        return np.zeros_like(values, dtype=float)
    return values.astype(float) / maximum


def paint_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_strokes: int,
    *,
    seed: int = 0,
    passes: tuple[PassConfig, ...] = DEFAULT_PASSES,
    residual_weight: float = 0.75,
    gradient_weight: float = 0.25,
) -> tuple[list[Stroke], tuple[int, int, int]]:
    """Build a stroke program using residual-guided coarse-to-fine passes.

    The canvas starts from a robust border-color estimate. After each pass, the
    current painting is rendered and compared against the target. New stroke
    centers are sampled from a blend of residual error and image-gradient
    magnitude, while orientation continues to follow local image structure.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette must have shape (K, 3)")
    if residual_weight < 0.0 or gradient_weight < 0.0:
        raise ValueError("sampling weights must be non-negative")
    if residual_weight + gradient_weight <= 0.0:
        raise ValueError("at least one sampling weight must be positive")

    h, w, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = _normalized_map(gradient_magnitude(image_rgb))
    counts = _allocate_counts(total_strokes, passes)

    strokes: list[Stroke] = []

    for pass_index, (config, count) in enumerate(zip(passes, counts, strict=True)):
        if count == 0:
            continue

        current = render_strokes(strokes, size=(w, h), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0

        residual = np.linalg.norm(image_rgb - current_rgb, axis=2)
        residual = _normalized_map(residual)

        weight_map = residual_weight * residual + gradient_weight * gradient
        weight_map += 1e-12

        strokes.extend(
            sample_gradient_strokes(
                image_rgb,
                palette,
                count,
                seed=seed + pass_index,
                weight_map=weight_map,
                min_length=config.min_length,
                max_length=config.max_length,
                width=config.width,
                opacity=config.opacity,
            )
        )

    return strokes, background
