"""Reconstruction metrics for Phase 0 experiments."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def reconstruction_metrics(target_rgb: np.ndarray, painted_rgb: np.ndarray) -> dict[str, float]:
    """Compute basic reconstruction metrics for two float RGB images in [0, 1]."""
    if target_rgb.shape != painted_rgb.shape:
        raise ValueError("target_rgb and painted_rgb must have the same shape")
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError("images must have shape (H, W, 3)")
    if not np.isfinite(target_rgb).all() or not np.isfinite(painted_rgb).all():
        raise ValueError("images must contain only finite values")

    mse = float(np.mean((target_rgb - painted_rgb) ** 2))
    psnr = float(peak_signal_noise_ratio(target_rgb, painted_rgb, data_range=1.0))
    ssim = float(structural_similarity(target_rgb, painted_rgb, channel_axis=2, data_range=1.0))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }
