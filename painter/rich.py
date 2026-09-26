"""Richer Phase 0F painter using area patches and tapered strokes."""

from __future__ import annotations

import math

import cv2
import numpy as np

from painter.background import estimate_border_background
from painter.iterative import normalize_map
from painter.renderer import render_strokes
from painter.sampling import gradient_magnitude, sample_gradient_strokes
from painter.stroke import (
    BezierRibbon,
    ClosedBezierRegion,
    EllipsePatch,
    PolygonPatch,
    Primitive,
    TaperedStroke,
)


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



def _polygon_mask(
    shape: tuple[int, int],
    patch: PolygonPatch,
) -> np.ndarray:
    """Rasterize one normalized polygon patch to a boolean mask."""
    height, width = shape
    points = np.asarray(
        [
            (
                round(vertex[0] * max(width - 1, 1)),
                round(vertex[1] * max(height - 1, 1)),
            )
            for vertex in patch.vertices
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _fit_polygon_to_component(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    opacity: float,
    max_vertices: int = 12,
) -> PolygonPatch | None:
    """Approximate one residual component with a compact contour polygon."""
    component = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8.0:
        return None

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.012 * perimeter)
    approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

    while len(approximation) > max_vertices:
        epsilon *= 1.35
        approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

    if len(approximation) < 3:
        return None

    height, width, _ = image_rgb.shape
    vertices = tuple(
        (
            float(np.clip(x / max(width - 1, 1), 0.0, 1.0)),
            float(np.clip(y / max(height - 1, 1), 0.0, 1.0)),
        )
        for x, y in approximation
    )

    ys, xs = np.nonzero(component)
    representative = np.median(image_rgb[ys, xs], axis=0)
    color = _nearest_color(representative, palette)
    return PolygonPatch(
        vertices=vertices,
        color=color,
        opacity=opacity,
    )


def _polygon_improves_error(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    patch: PolygonPatch,
    *,
    min_relative_improvement: float,
) -> tuple[bool, np.ndarray]:
    """Accept only polygon patches that reduce local raster error."""
    mask = _polygon_mask(current_rgb.shape[:2], patch)
    if not np.any(mask):
        return False, current_rgb

    before = float(np.mean((current_rgb[mask] - target_rgb[mask]) ** 2))
    color = np.asarray(patch.color, dtype=np.float32)
    candidate = current_rgb.copy()
    alpha = float(patch.opacity)
    candidate[mask] = (1.0 - alpha) * candidate[mask] + alpha * color
    after = float(np.mean((candidate[mask] - target_rgb[mask]) ** 2))
    return after < before * (1.0 - min_relative_improvement), candidate


def _fit_residual_polygons(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    *,
    background: tuple[int, int, int],
    max_patches: int,
    gradient: np.ndarray,
    opacity: float = 0.84,
) -> list[PolygonPatch]:
    """Fit compact polygons to high-residual, locally smooth image regions."""
    if max_patches <= 0:
        return []

    height, width, _ = image_rgb.shape
    current_rgb = np.empty_like(image_rgb, dtype=np.float32)
    current_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    smoothness = 1.0 - gradient
    patches: list[PolygonPatch] = []
    quantiles = (0.92, 0.86, 0.80, 0.72, 0.64, 0.56)

    for quantile in quantiles:
        if len(patches) >= max_patches:
            break

        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        score = residual * (0.15 + 0.85 * smoothness**1.4)
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
            candidates.append((float(score[labels == label].sum()), label))

        candidates.sort(reverse=True)
        for _component_score, label in candidates:
            if len(patches) >= max_patches:
                break
            patch = _fit_polygon_to_component(
                image_rgb,
                palette,
                labels,
                label,
                opacity=opacity,
            )
            if patch is None:
                continue
            accepted, candidate_rgb = _polygon_improves_error(
                current_rgb,
                image_rgb,
                patch,
                min_relative_improvement=0.008,
            )
            if accepted:
                patches.append(patch)
                current_rgb = candidate_rgb

    return patches


def paint_polygon_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    patch_fraction: float = 0.22,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Paint broad arbitrary regions with polygons and detail with tapered strokes."""
    if total_primitives < 1:
        raise ValueError("total_primitives must be positive")
    if not 0.0 <= patch_fraction < 1.0:
        raise ValueError("patch_fraction must lie in [0, 1)")

    height, width, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = normalize_map(gradient_magnitude(image_rgb))
    patch_budget = round(total_primitives * patch_fraction)
    patches = _fit_residual_polygons(
        image_rgb,
        palette,
        background=background,
        max_patches=patch_budget,
        gradient=gradient,
    )
    primitives: list[Primitive] = list(patches)

    stroke_count = total_primitives - len(primitives)
    if stroke_count:
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = 0.78 * residual + 0.22 * gradient
        base = sample_gradient_strokes(
            image_rgb,
            palette,
            stroke_count,
            seed=seed + 307,
            weight_map=weight_map,
            min_length=0.006,
            max_length=0.070,
            width=0.010,
            opacity=0.88,
        )
        primitives.extend(_to_tapered(base, seed=seed + 401))

    return primitives, background



def _ribbon_mask(
    shape: tuple[int, int],
    ribbon: BezierRibbon,
) -> np.ndarray:
    """Rasterize a ribbon mask through the canonical renderer."""
    height, width = shape
    image = render_strokes(
        [ribbon],
        size=(width, height),
        background=(0, 0, 0),
        samples_per_curve=48,
    )
    return np.asarray(image, dtype=np.uint8).max(axis=2) > 0


def _fit_ribbon_to_component(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    opacity: float,
) -> BezierRibbon | None:
    """Fit one cubic ribbon to an elongated connected residual component."""
    ys, xs = np.nonzero(labels == label)
    if len(xs) < 16:
        return None

    coords = np.column_stack((xs.astype(float), ys.astype(float)))
    center = coords.mean(axis=0)
    centered = coords - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_value = float(max(eigenvalues[order[0]], 1e-8))
    minor_value = float(max(eigenvalues[order[1]], 1e-8))
    if major_value / minor_value < 2.0:
        return None

    major = eigenvectors[:, order[0]]
    minor = eigenvectors[:, order[1]]
    longitudinal = centered @ major
    transverse = centered @ minor
    lo, hi = np.quantile(longitudinal, [0.03, 0.97])
    if hi - lo < 8.0:
        return None

    samples: list[np.ndarray] = []
    widths: list[float] = []
    positions = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    span = hi - lo
    for fraction in positions:
        anchor = lo + fraction * span
        window = np.abs(longitudinal - anchor) <= max(2.0, 0.12 * span)
        if np.count_nonzero(window) < 4:
            point = center + anchor * major
            half_width = 2.0 * math.sqrt(minor_value)
        else:
            local_long = float(np.median(longitudinal[window]))
            local_trans = float(np.median(transverse[window]))
            point = center + local_long * major + local_trans * minor
            half_width = float(
                np.quantile(np.abs(transverse[window] - local_trans), 0.85)
            )
        samples.append(point)
        widths.append(max(1.5, 2.0 * half_width))

    height, width, _ = image_rgb.shape

    def normalize(point: np.ndarray) -> tuple[float, float]:
        return (
            float(np.clip(point[0] / max(width - 1, 1), 0.0, 1.0)),
            float(np.clip(point[1] / max(height - 1, 1), 0.0, 1.0)),
        )

    representative = np.median(image_rgb[ys, xs], axis=0)
    scale = max(min(width, height), 1)
    width_start = float(np.clip(widths[0] / scale, 0.004, 0.18))
    width_mid = float(np.clip(max(widths[1], widths[2]) / scale, 0.004, 0.22))
    width_end = float(np.clip(widths[3] / scale, 0.004, 0.18))

    return BezierRibbon(
        p0=normalize(samples[0]),
        p1=normalize(samples[1]),
        p2=normalize(samples[2]),
        p3=normalize(samples[3]),
        width_start=width_start,
        width_mid=width_mid,
        width_end=width_end,
        color=_nearest_color(representative, palette),
        opacity=opacity,
    )


def _ribbon_improves_error(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    ribbon: BezierRibbon,
    *,
    min_relative_improvement: float,
) -> tuple[bool, np.ndarray]:
    mask = _ribbon_mask(current_rgb.shape[:2], ribbon)
    if not np.any(mask):
        return False, current_rgb
    before = float(np.mean((current_rgb[mask] - target_rgb[mask]) ** 2))
    color = np.asarray(ribbon.color, dtype=np.float32)
    candidate = current_rgb.copy()
    alpha = float(ribbon.opacity)
    candidate[mask] = (1.0 - alpha) * candidate[mask] + alpha * color
    after = float(np.mean((candidate[mask] - target_rgb[mask]) ** 2))
    return after < before * (1.0 - min_relative_improvement), candidate


def _fit_residual_ribbons(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    *,
    background: tuple[int, int, int],
    max_ribbons: int,
    opacity: float = 0.86,
) -> list[BezierRibbon]:
    """Fit ribbons to elongated connected residual masses."""
    if max_ribbons <= 0:
        return []

    height, width, _ = image_rgb.shape
    current_rgb = np.empty_like(image_rgb, dtype=np.float32)
    current_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    ribbons: list[BezierRibbon] = []

    for quantile in (0.90, 0.84, 0.76, 0.68, 0.58):
        if len(ribbons) >= max_ribbons:
            break
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        positive = residual[residual > 0.0]
        if positive.size == 0:
            break
        threshold = float(np.quantile(positive, quantile))
        binary = (residual >= threshold).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        candidates: list[tuple[float, int]] = []
        minimum_area = max(16, round(height * width * 0.00004))
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= minimum_area:
                candidates.append((float(residual[labels == label].sum()), label))
        candidates.sort(reverse=True)

        for _score, label in candidates:
            if len(ribbons) >= max_ribbons:
                break
            ribbon = _fit_ribbon_to_component(
                image_rgb,
                palette,
                labels,
                label,
                opacity=opacity,
            )
            if ribbon is None:
                continue
            accepted, candidate_rgb = _ribbon_improves_error(
                current_rgb,
                image_rgb,
                ribbon,
                min_relative_improvement=0.006,
            )
            if accepted:
                ribbons.append(ribbon)
                current_rgb = candidate_rgb

    return ribbons


def paint_mixed_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    ribbon_fraction: float = 0.10,
    polygon_fraction: float = 0.18,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Use ribbons for elongated masses, polygons for regions, strokes for detail."""
    if total_primitives < 1:
        raise ValueError("total_primitives must be positive")
    if ribbon_fraction < 0.0 or polygon_fraction < 0.0:
        raise ValueError("primitive fractions must be non-negative")
    if ribbon_fraction + polygon_fraction >= 1.0:
        raise ValueError("ribbon_fraction + polygon_fraction must be less than 1")

    height, width, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = normalize_map(gradient_magnitude(image_rgb))
    ribbon_budget = round(total_primitives * ribbon_fraction)
    polygon_budget = round(total_primitives * polygon_fraction)

    ribbons = _fit_residual_ribbons(
        image_rgb,
        palette,
        background=background,
        max_ribbons=ribbon_budget,
    )
    primitives: list[Primitive] = list(ribbons)

    current = render_strokes(primitives, size=(width, height), background=background)
    current_rgb = np.asarray(current, dtype=np.float32) / 255.0
    residual_after_ribbons = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))

    # Fit polygons to the remaining broad error rather than to the untouched target.
    polygon_image = image_rgb.copy()
    blend = residual_after_ribbons[..., None]
    background_rgb = np.asarray(background, dtype=np.float32) / 255.0
    polygon_image = blend * polygon_image + (1.0 - blend) * background_rgb
    polygons = _fit_residual_polygons(
        polygon_image,
        palette,
        background=background,
        max_patches=polygon_budget,
        gradient=gradient,
    )
    primitives.extend(polygons)

    stroke_count = total_primitives - len(primitives)
    if stroke_count > 0:
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = 0.76 * residual + 0.24 * gradient
        base = sample_gradient_strokes(
            image_rgb,
            palette,
            stroke_count,
            seed=seed + 503,
            weight_map=weight_map,
            min_length=0.006,
            max_length=0.065,
            width=0.009,
            opacity=0.86,
        )
        primitives.extend(_to_tapered(base, seed=seed + 607))

    return primitives, background



def adaptive_primitive_fractions(
    image_rgb: np.ndarray,
    *,
    background: tuple[int, int, int] | None = None,
) -> tuple[float, float]:
    """Choose ribbon and polygon budget fractions from residual topology.

    Large connected residual components increase broad-region allocation.
    Elongated components bias that broad budget toward ribbons; compact
    components bias it toward polygons.
    """
    if background is None:
        background = estimate_border_background(image_rgb)
    background_rgb = np.asarray(background, dtype=np.float32) / 255.0
    residual = np.linalg.norm(image_rgb - background_rgb, axis=2)
    positive = residual[residual > 0.02]
    if positive.size == 0:
        return 0.05, 0.10

    threshold = float(np.quantile(positive, 0.72))
    binary = (residual >= threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    total_area = image_rgb.shape[0] * image_rgb.shape[1]
    coherent_area = 0
    elongated_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(12, round(total_area * 0.00003)):
            continue
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 4:
            continue
        coords = np.column_stack((xs.astype(float), ys.astype(float)))
        covariance = np.cov(coords, rowvar=False)
        values = np.linalg.eigvalsh(covariance)
        minor = max(float(values[0]), 1e-8)
        major = max(float(values[1]), minor)
        coherent_area += area
        if major / minor >= 2.0:
            elongated_area += area

    coherent_fraction = min(coherent_area / max(total_area, 1), 0.45)
    broad_fraction = float(np.clip(0.16 + 1.35 * coherent_fraction, 0.18, 0.48))
    elongated_share = elongated_area / max(coherent_area, 1)

    ribbon_fraction = float(np.clip(broad_fraction * elongated_share, 0.04, 0.22))
    polygon_fraction = float(
        np.clip(broad_fraction - ribbon_fraction, 0.12, 0.34)
    )
    if ribbon_fraction + polygon_fraction > 0.55:
        scale = 0.55 / (ribbon_fraction + polygon_fraction)
        ribbon_fraction *= scale
        polygon_fraction *= scale
    return ribbon_fraction, polygon_fraction


def paint_adaptive_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Paint with an input-adaptive mix of ribbons, polygons, and detail strokes."""
    background = estimate_border_background(image_rgb)
    ribbon_fraction, polygon_fraction = adaptive_primitive_fractions(
        image_rgb,
        background=background,
    )
    return paint_mixed_rich_residual(
        image_rgb,
        palette,
        total_primitives,
        seed=seed,
        ribbon_fraction=ribbon_fraction,
        polygon_fraction=polygon_fraction,
    )



def _normalize_point_xy(
    point: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        float(np.clip(point[0] / max(width - 1, 1), 0.0, 1.0)),
        float(np.clip(point[1] / max(height - 1, 1), 0.0, 1.0)),
    )


def _catmull_rom_closed_segments(
    vertices: np.ndarray,
    *,
    width: int,
    height: int,
    tension: float = 1.0,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Convert a closed polygon into C1-smooth cubic Bezier segments."""
    count = len(vertices)
    segments = []
    for index in range(count):
        p_prev = vertices[(index - 1) % count]
        p0 = vertices[index]
        p3 = vertices[(index + 1) % count]
        p_next = vertices[(index + 2) % count]

        p1 = p0 + tension * (p3 - p_prev) / 6.0
        p2 = p3 - tension * (p_next - p0) / 6.0

        segment = (
            _normalize_point_xy(p0, width=width, height=height),
            _normalize_point_xy(p1, width=width, height=height),
            _normalize_point_xy(p2, width=width, height=height),
            _normalize_point_xy(p3, width=width, height=height),
        )
        segments.append(segment)
    return tuple(segments)


def _fit_closed_bezier_region_to_component(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    opacity: float,
    max_vertices: int = 10,
) -> ClosedBezierRegion | None:
    """Fit a smooth closed Bezier region to one connected residual component."""
    component = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 16.0:
        return None

    perimeter = max(cv2.arcLength(contour, True), 1.0)
    epsilon = 0.01 * perimeter
    approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    while len(approximation) > max_vertices:
        epsilon *= 1.3
        approximation = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)

    if len(approximation) < 4:
        return None

    height, width, _ = image_rgb.shape
    ys, xs = np.nonzero(component)
    representative = np.median(image_rgb[ys, xs], axis=0)

    return ClosedBezierRegion(
        segments=_catmull_rom_closed_segments(
            approximation,
            width=width,
            height=height,
            tension=0.72,
        ),
        color=_nearest_color(representative, palette),
        opacity=opacity,
    )


def _closed_region_improves_error(
    current_rgb: np.ndarray,
    target_rgb: np.ndarray,
    region: ClosedBezierRegion,
    *,
    min_relative_improvement: float,
) -> tuple[bool, np.ndarray]:
    height, width, _ = current_rgb.shape
    overlay = render_strokes(
        [region],
        size=(width, height),
        background=(0, 0, 0),
        samples_per_curve=24,
    )
    mask = np.asarray(overlay, dtype=np.uint8).max(axis=2) > 0
    if not np.any(mask):
        return False, current_rgb

    before = float(np.mean((current_rgb[mask] - target_rgb[mask]) ** 2))
    color = np.asarray(region.color, dtype=np.float32)
    alpha = float(region.opacity)
    candidate = current_rgb.copy()
    candidate[mask] = (1.0 - alpha) * candidate[mask] + alpha * color
    after = float(np.mean((candidate[mask] - target_rgb[mask]) ** 2))
    return after < before * (1.0 - min_relative_improvement), candidate


def _fit_residual_closed_regions(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    *,
    background: tuple[int, int, int],
    max_regions: int,
    opacity: float = 0.86,
) -> list[ClosedBezierRegion]:
    """Fit smooth closed regions to large coherent residual components."""
    if max_regions <= 0:
        return []

    height, width, _ = image_rgb.shape
    current_rgb = np.empty_like(image_rgb, dtype=np.float32)
    current_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    regions: list[ClosedBezierRegion] = []

    for quantile in (0.88, 0.80, 0.70, 0.60):
        if len(regions) >= max_regions:
            break

        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        positive = residual[residual > 0.0]
        if positive.size == 0:
            break
        threshold = float(np.quantile(positive, quantile))
        binary = (residual >= threshold).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        candidates: list[tuple[float, int]] = []
        minimum_area = max(24, round(height * width * 0.00008))
        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= minimum_area:
                candidates.append((float(residual[labels == label].sum()), label))
        candidates.sort(reverse=True)

        for _score, label in candidates:
            if len(regions) >= max_regions:
                break
            region = _fit_closed_bezier_region_to_component(
                image_rgb,
                palette,
                labels,
                label,
                opacity=opacity,
            )
            if region is None:
                continue
            accepted, candidate_rgb = _closed_region_improves_error(
                current_rgb,
                image_rgb,
                region,
                min_relative_improvement=0.008,
            )
            if accepted:
                regions.append(region)
                current_rgb = candidate_rgb

    return regions


def paint_closed_region_rich_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    closed_region_fraction: float = 0.16,
    ribbon_fraction: float = 0.08,
    polygon_fraction: float = 0.12,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Use smooth closed regions first, then ribbons/polygons, then detail strokes."""
    if total_primitives < 1:
        raise ValueError("total_primitives must be positive")
    total_fraction = closed_region_fraction + ribbon_fraction + polygon_fraction
    if min(closed_region_fraction, ribbon_fraction, polygon_fraction) < 0.0:
        raise ValueError("primitive fractions must be non-negative")
    if total_fraction >= 1.0:
        raise ValueError("broad primitive fractions must sum to less than 1")

    height, width, _ = image_rgb.shape
    background = estimate_border_background(image_rgb)
    gradient = normalize_map(gradient_magnitude(image_rgb))

    region_budget = round(total_primitives * closed_region_fraction)
    ribbon_budget = round(total_primitives * ribbon_fraction)
    polygon_budget = round(total_primitives * polygon_fraction)

    regions = _fit_residual_closed_regions(
        image_rgb,
        palette,
        background=background,
        max_regions=region_budget,
    )
    primitives: list[Primitive] = list(regions)

    current = render_strokes(primitives, size=(width, height), background=background)
    current_rgb = np.asarray(current, dtype=np.float32) / 255.0
    residual_after_regions = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
    residual_image = (
        residual_after_regions[..., None] * image_rgb
        + (1.0 - residual_after_regions[..., None])
        * (np.asarray(background, dtype=np.float32) / 255.0)
    )

    ribbons = _fit_residual_ribbons(
        residual_image,
        palette,
        background=background,
        max_ribbons=ribbon_budget,
    )
    primitives.extend(ribbons)

    current = render_strokes(primitives, size=(width, height), background=background)
    current_rgb = np.asarray(current, dtype=np.float32) / 255.0
    residual_after_ribbons = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
    polygon_image = (
        residual_after_ribbons[..., None] * image_rgb
        + (1.0 - residual_after_ribbons[..., None])
        * (np.asarray(background, dtype=np.float32) / 255.0)
    )
    polygons = _fit_residual_polygons(
        polygon_image,
        palette,
        background=background,
        max_patches=polygon_budget,
        gradient=gradient,
    )
    primitives.extend(polygons)

    remaining = total_primitives - len(primitives)
    if remaining > 0:
        current = render_strokes(primitives, size=(width, height), background=background)
        current_rgb = np.asarray(current, dtype=np.float32) / 255.0
        residual = normalize_map(np.linalg.norm(image_rgb - current_rgb, axis=2))
        weight_map = 0.72 * residual + 0.28 * gradient
        base = sample_gradient_strokes(
            image_rgb,
            palette,
            remaining,
            seed=seed + 701,
            weight_map=weight_map,
            min_length=0.006,
            max_length=0.060,
            width=0.008,
            opacity=0.84,
        )
        primitives.extend(_to_tapered(base, seed=seed + 809))

    return primitives, background


def paint_adaptive_closed_region_residual(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
) -> tuple[list[Primitive], tuple[int, int, int]]:
    """Adapt broad-region budget while reserving smooth closed-shape capacity."""
    background = estimate_border_background(image_rgb)
    ribbon_fraction, polygon_fraction = adaptive_primitive_fractions(
        image_rgb,
        background=background,
    )
    broad_total = min(0.52, ribbon_fraction + polygon_fraction + 0.12)
    closed_fraction = float(np.clip(0.10 + 0.45 * broad_total, 0.12, 0.24))
    remaining_broad = max(0.08, broad_total - closed_fraction)
    scale = remaining_broad / max(ribbon_fraction + polygon_fraction, 1e-8)
    return paint_closed_region_rich_residual(
        image_rgb,
        palette,
        total_primitives,
        seed=seed,
        closed_region_fraction=closed_fraction,
        ribbon_fraction=ribbon_fraction * scale,
        polygon_fraction=polygon_fraction * scale,
    )
