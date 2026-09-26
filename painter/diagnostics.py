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
    target_edge_energy = float(np.mean(np.hypot(target_gx, target_gy)))
    painted_edge_energy = float(np.mean(np.hypot(painted_gx, painted_gy)))
    edge_energy_ratio = painted_edge_energy / max(target_edge_energy, 1e-8)

    target_laplacian = cv2.Laplacian(target_gray, cv2.CV_32F)
    painted_laplacian = cv2.Laplacian(painted_gray, cv2.CV_32F)
    high_frequency_ratio = float(
        np.mean(np.abs(painted_laplacian))
        / max(float(np.mean(np.abs(target_laplacian))), 1e-8)
    )

    residual = np.linalg.norm(target_rgb - painted_rgb, axis=2)
    residual_threshold = max(float(np.quantile(residual, 0.85)), 0.03)
    residual_mask = (residual >= residual_threshold).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        residual_mask,
        connectivity=8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([])
    large_component_fraction = (
        float(component_areas.max() / residual_mask.size)
        if component_areas.size
        else 0.0
    )

    return {
        "foreground_iou": foreground_iou,
        "boundary_f1": boundary_f1,
        "mean_boundary_distance_px": mean_boundary_distance,
        "edge_energy_ratio": float(edge_energy_ratio),
        "high_frequency_ratio": high_frequency_ratio,
        "residual_component_count": max(component_count - 1, 0),
        "largest_residual_component_fraction": large_component_fraction,
        "foreground_fraction_target": float(np.mean(target_fg)),
        "foreground_fraction_painted": float(np.mean(painted_fg)),
    }
