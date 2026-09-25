"""Richer Phase 0F painter using area patches and tapered strokes."""

from __future__ import annotations

import math

import cv2
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

    patch_count = round(total_primitives * patch_fraction)
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



def _ellipse_mask(
    shape: tuple[int, int],
    patch: EllipsePatch,
) -> np.ndarray:
    """Rasterize one ellipse patch to a boolean mask."""
    height, width = shape
    center = (
        round(patch.center[0] * max(width - 1, 1)),
        round(patch.center[1] * max(height - 1, 1)),
    )
    axes = (
        max(1, round(patch.radius_x * max(width - 1, 1))),
        max(1, round(patch.radius_y * max(height - 1, 1))),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        center,
        axes,
        math.degrees(patch.angle),
        0.0,
        360.0,
        1,
        thickness=-1,
    )
    return mask.astype(bool)


def _fit_patch_to_component(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    opacity: float,
) -> EllipsePatch | None:
    """Fit an ellipse to one connected residual component."""
    ys, xs = np.nonzero(labels == label)
    if len(xs) < 8:
        return None

    coords = np.column_stack((xs.astype(float), ys.astype(float)))
    center_px = coords.mean(axis=0)
    centered = coords - center_px
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # For points uniformly filling an ellipse, covariance is approximately r^2 / 4.
    radius_major = max(1.0, 2.0 * math.sqrt(max(float(eigenvalues[0]), 1e-8)))
    radius_minor = max(1.0, 2.0 * math.sqrt(max(float(eigenvalues[1]), 1e-8)))
    major_vector = eigenvectors[:, 0]
    angle = math.atan2(float(major_vector[1]), float(major_vector[0]))

    height, width, _ = image_rgb.shape
    center = (
        float(np.clip(center_px[0] / max(width - 1, 1), 0.0, 1.0)),
        float(np.clip(center_px[1] / max(height - 1, 1), 0.0, 1.0)),
    )
    radius_x = float(np.clip(radius_major / max(width - 1, 1), 0.002, 0.18))
    radius_y = float(np.clip(radius_minor / max(height - 1, 1), 0.002, 0.18))

    component_pixels = image_rgb[ys, xs]
    representative = np.median(component_pixels, axis=0)
    color = _nearest_color(representative, palette)

    return EllipsePatch(
        center=center,
        radius_x=radius_x,
        radius_y=radius_y,
        angle=angle,
        color=color,
        opacity=opacity,
    )


def _patch_improves_error(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    patch: EllipsePatch,
    *,
    min_relative_improvement: float,
) -> tuple[bool, np.ndarray]:
    """Return whether a patch lowers RGB squared error and its composited canvas."""
    mask = _ellipse_mask(current_rgb.shape[:2], patch)
    if not np.any(mask):
        return False, current_rgb

    before = float(np.mean((current_rgb[mask] - target_rgb[mask]) ** 2))
    color = np.asarray(patch.color, dtype=np.float32)
    candidate = current_rgb.copy()
    alpha = float(patch.opacity)
    candidate[mask] = (1.0 - alpha) * candidate[mask] + alpha * color
    after = float(np.mean((candidate[mask] - target_rgb[mask]) ** 2))

    threshold = before * (1.0 - min_relative_improvement)
    return after < threshold, candidate


def _fit_residual_patches(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    *,
    background: tuple[int, int, int],
    max_patches: int,
    gradient: np.ndarray,
    opacity: float = 0.82,
) -> list[EllipsePatch]:
    """Fit accepted ellipses to connected smooth residual regions."""
    if max_patches <= 0:
        return []

    height, width, _ = image_rgb.shape
    current_rgb = np.empty_like(image_rgb, dtype=np.float32)
    current_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    smoothness = 1.0 - gradient
    patches: list[EllipsePatch] = []
    quantiles = (0.90, 0.84, 0.78, 0.70, 0.62, 0.54)

    for quantile in quantiles:
        if len(patches) >= max_patches:
            break

        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        score = residual * (0.10 + 0.90 * smoothness**1.6)
        positive = score[score > 0.0]
        if positive.size == 0:
            break

        threshold = float(np.quantile(positive, quantile))
        binary = (score >= threshold).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        candidates: list[tuple[float, int]] = []
        minimum_area = max(8, round(height * width * 0.00002))
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component_score = float(score[labels == label].sum())
            candidates.append((component_score, label))

        candidates.sort(reverse=True)
        for _component_score, label in candidates:
            if len(patches) >= max_patches:
                break
            patch = _fit_patch_to_component(
                image_rgb,
                palette,
                labels,
                label,
                opacity=opacity,
            )
            if patch is None:
                continue

            accepted, candidate_rgb = _patch_improves_error(
                current_rgb,
                image_rgb,
                patch,
                min_relative_improvement=0.01,
            )
            if not accepted:
                continue

            patches.append(patch)
            current_rgb = candidate_rgb

    return patches


def paint_region_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    patch_fraction: float = 0.18,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Paint with fitted residual-region patches plus tapered detail strokes."""
    if total_primitives < 1:
        raise ValueError("total_primitives must be positive")
    if not 0.0 <= patch_fraction < 1.0:
        raise ValueError("patch_fraction must lie in [0, 1)")

    height, width, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = normalize_map(gradient_magnitude(image_rgb))
    patch_budget = round(total_primitives * patch_fraction)

    patches = _fit_residual_patches(
        image_rgb,
        palette,
        background=background,
        max_patches=patch_budget,
        gradient=gradient,
    )
    primitives: list[Primitive] = list(patches)

    # Unused patch budget is returned to tapered detail strokes.
    stroke_count = total_primitives - len(primitives)
    if stroke_count:
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = 0.82 * residual + 0.18 * gradient
        base = sample_gradient_strokes(
            image_rgb,
            palette,
            stroke_count,
            seed=seed + 101,
            weight_map=weight_map,
            min_length=0.006,
            max_length=0.065,
            width=0.010,
            opacity=0.88,
        )
        primitives.extend(_to_tapered(base, seed=seed + 211))

    return primitives, background
