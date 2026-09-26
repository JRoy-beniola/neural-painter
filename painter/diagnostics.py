"""Image-level diagnostics for self-directed painter experiments."""

from __future__ import annotations

import cv2
import numpy as np


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32) / 255.0


def _foreground_mask(
    image_rgb: np.ndarray,
    background_rgb: tuple[int, int, int],
    *,
    threshold: float = 0.08,
) -> np.ndarray:
    background = np.asarray(background_rgb, dtype=np.float32) / 255.0
    distance = np.linalg.norm(image_rgb - background, axis=2)
    return distance > threshold


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 - eroded) > 0


def _distance_to(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 5)


def _energy_ratio(
    painted_energy: np.ndarray,
    target_energy: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 1.0
    numerator = float(np.mean(painted_energy[mask]))
    denominator = float(np.mean(target_energy[mask]))
    if denominator < 1e-8:
        return 1.0 if numerator < 1e-8 else numerator / 1e-8
    return numerator / denominator


def _residual_topology(
    residual: np.ndarray,
) -> dict[str, float | int]:
    """Measure meaningful residual regions without percentile-forced area."""
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(0.06, median + 2.5 * robust_sigma)

    residual_mask = residual >= threshold
    residual_pixel_fraction = float(np.mean(residual_mask))
    residual_energy = residual * residual
    total_residual_energy = float(np.sum(residual_energy))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual_mask.astype(np.uint8),
        connectivity=8,
    )

    largest_area_fraction = 0.0
    largest_energy_share = 0.0
    largest_eccentricity = 1.0
    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / residual.size
        energy_share = (
            float(np.sum(residual_energy[component])) / total_residual_energy
            if total_residual_energy > 1e-12
            else 0.0
        )
        if area_fraction > largest_area_fraction:
            largest_area_fraction = float(area_fraction)

        if energy_share > largest_energy_share:
            largest_energy_share = float(energy_share)
            ys, xs = np.nonzero(component)
            if len(xs) >= 4:
                coords = np.column_stack((xs.astype(float), ys.astype(float)))
                covariance = np.cov(coords, rowvar=False)
                values = np.linalg.eigvalsh(covariance)
                minor = max(float(values[0]), 1e-8)
                major = max(float(values[1]), minor)
                largest_eccentricity = float(np.sqrt(major / minor))

    return {
        "residual_threshold": threshold,
        "residual_pixel_fraction": residual_pixel_fraction,
        "residual_component_count": max(component_count - 1, 0),
        "largest_residual_component_fraction": largest_area_fraction,
        "largest_residual_component_energy_share": largest_energy_share,
        "largest_residual_component_eccentricity": largest_eccentricity,
    }


def image_diagnostics(
    target_rgb: np.ndarray,
    painted_rgb: np.ndarray,
    *,
    background_rgb: tuple[int, int, int],
) -> dict[str, float | int]:
    """Measure structural failure modes that MSE/SSIM alone cannot explain."""
    if target_rgb.shape != painted_rgb.shape:
        raise ValueError("target_rgb and painted_rgb must have the same shape")

    target_fg = _foreground_mask(target_rgb, background_rgb)
    painted_fg = _foreground_mask(painted_rgb, background_rgb)
    intersection = np.count_nonzero(target_fg & painted_fg)
    union = np.count_nonzero(target_fg | painted_fg)
    foreground_iou = float(intersection / union) if union else 1.0

    target_boundary = _boundary(target_fg)
    painted_boundary = _boundary(painted_fg)
    target_distance = _distance_to(target_boundary)
    painted_distance = _distance_to(painted_boundary)

    painted_boundary_count = max(np.count_nonzero(painted_boundary), 1)
    target_boundary_count = max(np.count_nonzero(target_boundary), 1)
    mean_boundary_distance = float(
        (
            target_distance[painted_boundary].sum() / painted_boundary_count
            + painted_distance[target_boundary].sum() / target_boundary_count
        )
        * 0.5
    )

    tolerance = 2.0
    precision = float(
        np.count_nonzero(painted_boundary & (target_distance <= tolerance))
        / painted_boundary_count
    )
    recall = float(
        np.count_nonzero(target_boundary & (painted_distance <= tolerance))
        / target_boundary_count
    )
    boundary_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    target_gray = _gray(target_rgb)
    painted_gray = _gray(painted_rgb)
    target_gx = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3)
    target_gy = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    painted_gx = cv2.Sobel(painted_gray, cv2.CV_32F, 1, 0, ksize=3)
    painted_gy = cv2.Sobel(painted_gray, cv2.CV_32F, 0, 1, ksize=3)
    target_edge = np.hypot(target_gx, target_gy)
    painted_edge = np.hypot(painted_gx, painted_gy)
    edge_energy_ratio = float(np.mean(painted_edge)) / max(
        float(np.mean(target_edge)),
        1e-8,
    )

    target_laplacian = np.abs(cv2.Laplacian(target_gray, cv2.CV_32F))
    painted_laplacian = np.abs(cv2.Laplacian(painted_gray, cv2.CV_32F))
    high_frequency_ratio = float(np.mean(painted_laplacian)) / max(
        float(np.mean(target_laplacian)),
        1e-8,
    )

    boundary_band = cv2.dilate(
        target_boundary.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    interior = target_fg & ~boundary_band
    exterior = ~target_fg & ~boundary_band

    residual = np.linalg.norm(target_rgb - painted_rgb, axis=2)
    topology = _residual_topology(residual)

    return {
        "foreground_iou": foreground_iou,
        "boundary_f1": boundary_f1,
        "mean_boundary_distance_px": mean_boundary_distance,
        "edge_energy_ratio": edge_energy_ratio,
        "high_frequency_ratio": high_frequency_ratio,
        "high_frequency_boundary_ratio": _energy_ratio(
            painted_laplacian,
            target_laplacian,
            boundary_band,
        ),
        "high_frequency_interior_ratio": _energy_ratio(
            painted_laplacian,
            target_laplacian,
            interior,
        ),
        "high_frequency_exterior_ratio": _energy_ratio(
            painted_laplacian,
            target_laplacian,
            exterior,
        ),
        **topology,
        "foreground_fraction_target": float(np.mean(target_fg)),
        "foreground_fraction_painted": float(np.mean(painted_fg)),
    }
