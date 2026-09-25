"""Structured residual painter with separate edge and interior policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from painter.background import estimate_border_background
from painter.iterative import _allocate_counts, _normalized_map
from painter.renderer import render_strokes
from painter.sampling import gradient_magnitude, sample_fill_strokes, sample_gradient_strokes
from painter.stroke import Stroke


@dataclass(frozen=True, slots=True)
class StructuredPass:
    """Stroke schedule for one structured residual pass."""

    fraction: float
    edge_fraction: float
    edge_min_length: float
    edge_max_length: float
    edge_width: float
    fill_min_length: float
    fill_max_length: float
    fill_width: float


DEFAULT_STRUCTURED_PASSES = (
    StructuredPass(0.20, 0.35, 0.060, 0.180, 0.018, 0.050, 0.160, 0.040),
    StructuredPass(0.30, 0.45, 0.035, 0.120, 0.012, 0.035, 0.120, 0.028),
    StructuredPass(0.50, 0.60, 0.015, 0.075, 0.007, 0.020, 0.080, 0.016),
)


def _pass_counts(total_strokes: int) -> list[int]:
    from painter.iterative import PassConfig

    proxy = tuple(
        PassConfig(config.fraction, 0.01, 0.02, 0.01, 1.0)
        for config in DEFAULT_STRUCTURED_PASSES
    )
    return _allocate_counts(total_strokes, proxy)


def paint_structured_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_strokes: int,
    *,
    seed: int = 0,
    edge_power: float = 1.5,
    residual_power: float = 1.0,
) -> tuple[list[Stroke], tuple[int, int, int]]:
    """Paint with explicit contour and smooth-interior stroke policies."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) == 0:
        raise ValueError("palette must have shape (K, 3)")
    if edge_power <= 0.0 or residual_power <= 0.0:
        raise ValueError("edge_power and residual_power must be positive")

    h, w, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    edge_strength = _normalized_map(gradient_magnitude(image_rgb))
    counts = _pass_counts(total_strokes)

    strokes: list[Stroke] = []

    for pass_index, (config, count) in enumerate(
        zip(DEFAULT_STRUCTURED_PASSES, counts, strict=True)
    ):
        if count == 0:
            continue

        current = render_strokes(strokes, size=(w, h), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = _normalized_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        residual_term = np.power(residual, residual_power)

        edge_count = int(round(count * config.edge_fraction))
        edge_count = min(max(edge_count, 0), count)
        fill_count = count - edge_count

        edge_weights = residual_term * np.power(edge_strength, edge_power)
        fill_weights = residual_term * np.power(1.0 - edge_strength, edge_power)

        if edge_count:
            strokes.extend(
                sample_gradient_strokes(
                    image_rgb,
                    palette,
                    edge_count,
                    seed=seed + pass_index * 2,
                    weight_map=edge_weights + 1e-12,
                    min_length=config.edge_min_length,
                    max_length=config.edge_max_length,
                    width=config.edge_width,
                    opacity=0.90,
                )
            )

        if fill_count:
            strokes.extend(
                sample_fill_strokes(
                    image_rgb,
                    palette,
                    fill_count,
                    seed=seed + pass_index * 2 + 1,
                    weight_map=fill_weights + 1e-12,
                    min_length=config.fill_min_length,
                    max_length=config.fill_max_length,
                    width=config.fill_width,
                    opacity=0.72,
                )
            )

    return strokes, background
