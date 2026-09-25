"""Richer Phase 0F painter using area patches and tapered strokes."""

from __future__ import annotations

import math

import numpy as np

from painter.background import estimate_border_background
from painter.iterative import normalize_map
from painter.renderer import render_strokes
from painter.sampling import gradient_magnitude, sample_gradient_strokes
from painter.stroke import EllipsePatch, Primitive, TaperedStroke


def _nearest_color(pixel: np.ndarray, palette: np.ndarray) -> tuple[float, float, float]:
    index = int(np.argmin(np.sum((palette - pixel) ** 2, axis=1)))
    return tuple(float(value) for value in palette[index])


def _sample_patches(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    count: int,
    *,
    seed: int,
    weight_map: np.ndarray,
    radius_x: tuple[float, float],
    radius_y: tuple[float, float],
    opacity: float,
) -> list[EllipsePatch]:
    if count <= 0:
        return []

    weights = weight_map.astype(float, copy=True)
    weights += 1e-12
    flat_weights = weights.ravel()
    flat_weights /= flat_weights.sum()

    rng = np.random.default_rng(seed)
    height, width, _ = image_rgb.shape
    indices = rng.choice(height * width, size=count, replace=True, p=flat_weights)

    patches: list[EllipsePatch] = []
    for flat in indices:
        y, x = divmod(int(flat), width)
        center = (
            x / max(width - 1, 1),
            y / max(height - 1, 1),
        )
        patches.append(
            EllipsePatch(
                center=center,
                radius_x=float(rng.uniform(*radius_x)),
                radius_y=float(rng.uniform(*radius_y)),
                angle=float(rng.uniform(0.0, math.pi)),
                color=_nearest_color(image_rgb[y, x], palette),
                opacity=opacity,
            )
        )
    return patches


def _to_tapered(
    strokes,
    *,
    seed: int,
) -> list[TaperedStroke]:
    rng = np.random.default_rng(seed)
    tapered: list[TaperedStroke] = []
    for stroke in strokes:
        start_scale = float(rng.uniform(0.22, 0.55))
        end_scale = float(rng.uniform(0.22, 0.55))
        mid_scale = float(rng.uniform(1.0, 1.35))
        tapered.append(
            TaperedStroke(
                p0=stroke.p0,
                p1=stroke.p1,
                p2=stroke.p2,
                width_start=max(0.001, stroke.width * start_scale),
                width_mid=min(1.0, stroke.width * mid_scale),
                width_end=max(0.001, stroke.width * end_scale),
                color=stroke.color,
                opacity=stroke.opacity,
            )
        )
    return tapered


def paint_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    patch_fraction: float = 0.30,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Paint with broad ellipse patches followed by tapered detail strokes.

    Broad low-gradient residual regions are first explained with filled patches.
    Remaining error is then addressed using tapered Bezier strokes aligned with
    local image structure.
    """
    if total_primitives < 1:
        raise ValueError("total_primitives must be positive")
    if not 0.0 <= patch_fraction < 1.0:
        raise ValueError("patch_fraction must lie in [0, 1)")

    height, width, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = normalize_map(gradient_magnitude(image_rgb))
    smoothness = 1.0 - gradient

    patch_count = int(round(total_primitives * patch_fraction))
    stroke_count = total_primitives - patch_count
    coarse_count = patch_count // 2
    medium_count = patch_count - coarse_count

    primitives: list[Primitive] = []

    for stage, (count, rx, ry, opacity) in enumerate(
        (
            (coarse_count, (0.035, 0.095), (0.025, 0.070), 0.78),
            (medium_count, (0.018, 0.055), (0.012, 0.040), 0.74),
        )
    ):
        if count == 0:
            continue
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = residual * (0.15 + 0.85 * smoothness**1.5)
        primitives.extend(
            _sample_patches(
                image_rgb,
                palette,
                count,
                seed=seed + stage,
                weight_map=weight_map,
                radius_x=rx,
                radius_y=ry,
                opacity=opacity,
            )
        )

    if stroke_count:
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = 0.75 * residual + 0.25 * gradient
        base = sample_gradient_strokes(
            image_rgb,
            palette,
            stroke_count,
            seed=seed + 17,
            weight_map=weight_map,
            min_length=0.008,
            max_length=0.075,
            width=0.012,
            opacity=0.86,
        )
        primitives.extend(_to_tapered(base, seed=seed + 31))

    return primitives, background
