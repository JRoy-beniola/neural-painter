"""Small differentiable soft rasterizer used for local stroke refinement."""

from __future__ import annotations

import torch


def render_soft_strokes(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    base_rgb: torch.Tensor,
    samples_per_curve: int = 10,
) -> torch.Tensor:
    """Render selected quadratic Bezier strokes over a fixed RGB base image.

    All coordinates and colors are normalized to [0, 1]. The renderer uses
    Gaussian coverage around sampled points on each Bezier curve. It is not a
    physical brush model; its role is to provide usable gradients for Phase 0
    local refinement.
    """
    if base_rgb.ndim != 3 or base_rgb.shape[-1] != 3:
        raise ValueError("base_rgb must have shape (H, W, 3)")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    device = p0.device
    dtype = p0.dtype
    height, width, _ = base_rgb.shape

    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)

    t = torch.linspace(0.0, 1.0, samples_per_curve, device=device, dtype=dtype)
    one_minus_t = 1.0 - t
    points = (
        one_minus_t[None, :, None] ** 2 * p0[:, None, :]
        + 2.0 * one_minus_t[None, :, None] * t[None, :, None] * p1[:, None, :]
        + t[None, :, None] ** 2 * p2[:, None, :]
    )

    diff = grid[None, :, :, None, :] - points[:, None, None, :, :]
    distance_sq = torch.sum(diff * diff, dim=-1).amin(dim=-1)

    sigma = torch.clamp(widths[:, None, None] * 0.5, min=1e-4)
    alpha = opacities[:, None, None] * torch.exp(-distance_sq / (2.0 * sigma * sigma))
    alpha = torch.clamp(alpha, 0.0, 0.98)

    coverage = 1.0 - torch.prod(1.0 - alpha, dim=0)
    weighted = torch.sum(alpha[..., None] * colors[:, None, None, :], dim=0)
    weight_sum = torch.sum(alpha, dim=0)[..., None].clamp_min(1e-6)
    stroke_rgb = weighted / weight_sum

    return base_rgb * (1.0 - coverage[..., None]) + stroke_rgb * coverage[..., None]
