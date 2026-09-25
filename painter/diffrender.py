"""Small differentiable soft rasterizer used for local stroke refinement."""

from __future__ import annotations

import torch


def _ordered_composite(
    base_rgb: torch.Tensor,
    alpha: torch.Tensor,
    colors: torch.Tensor,
) -> torch.Tensor:
    """Alpha-composite primitives in program order like the raster renderer."""
    canvas = base_rgb
    for index in range(alpha.shape[0]):
        a = alpha[index][..., None]
        canvas = canvas * (1.0 - a) + colors[index][None, None, :] * a
    return canvas



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



def soft_tapered_alpha_maps(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    width_start: torch.Tensor,
    width_mid: torch.Tensor,
    width_end: torch.Tensor,
    opacities: torch.Tensor,
    *,
    image_size: tuple[int, int],
    samples_per_curve: int = 24,
    edge_softness_pixels: float = 0.75,
) -> torch.Tensor:
    """Return one soft alpha map per tapered stroke."""
    width, height = image_size
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    device = p0.device
    dtype = p0.dtype
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)

    t = torch.linspace(0.0, 1.0, samples_per_curve, device=device, dtype=dtype)
    omt = 1.0 - t
    points = (
        omt[None, :, None] ** 2 * p0[:, None, :]
        + 2.0 * omt[None, :, None] * t[None, :, None] * p1[:, None, :]
        + t[None, :, None] ** 2 * p2[:, None, :]
    )

    first_half = torch.clamp(t * 2.0, 0.0, 1.0)
    second_half = torch.clamp((t - 0.5) * 2.0, 0.0, 1.0)
    widths_first = (
        (1.0 - first_half[None, :]) * width_start[:, None]
        + first_half[None, :] * width_mid[:, None]
    )
    widths_second = (
        (1.0 - second_half[None, :]) * width_mid[:, None]
        + second_half[None, :] * width_end[:, None]
    )
    sample_widths = torch.where(t[None, :] <= 0.5, widths_first, widths_second)

    diff = grid[None, :, :, None, :] - points[:, None, None, :, :]
    distance = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-10)
    radius = 0.5 * sample_widths[:, None, None, :]
    pixel_softness = edge_softness_pixels / max(min(height, width) - 1, 1)
    sample_coverage = torch.sigmoid((radius - distance) / pixel_softness)
    coverage = 1.0 - torch.prod(1.0 - sample_coverage, dim=-1)
    return torch.clamp(opacities[:, None, None] * coverage, 0.0, 1.0)


def soft_ellipse_alpha_maps(
    centers: torch.Tensor,
    radii_x: torch.Tensor,
    radii_y: torch.Tensor,
    angles: torch.Tensor,
    opacities: torch.Tensor,
    *,
    image_size: tuple[int, int],
    edge_softness_pixels: float = 0.75,
) -> torch.Tensor:
    """Return one soft alpha map per oriented ellipse patch."""
    width, height = image_size
    device = centers.device
    dtype = centers.dtype
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dx = grid_x[None, :, :] - centers[:, 0, None, None]
    dy = grid_y[None, :, :] - centers[:, 1, None, None]

    cos_a = torch.cos(angles)[:, None, None]
    sin_a = torch.sin(angles)[:, None, None]
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy

    rx = radii_x[:, None, None].clamp_min(1e-4)
    ry = radii_y[:, None, None].clamp_min(1e-4)
    normalized_radius = torch.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2 + 1e-10)
    signed_distance = (1.0 - normalized_radius) * torch.minimum(rx, ry)
    pixel_softness = edge_softness_pixels / max(min(height, width) - 1, 1)
    coverage = torch.sigmoid(signed_distance / pixel_softness)
    return torch.clamp(opacities[:, None, None] * coverage, 0.0, 1.0)


def render_soft_tapered_strokes(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    width_start: torch.Tensor,
    width_mid: torch.Tensor,
    width_end: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    base_rgb: torch.Tensor,
    samples_per_curve: int = 24,
    edge_softness_pixels: float = 0.75,
) -> torch.Tensor:
    """Render tapered Bezier strokes using near-raster hard-edge soft coverage."""
    if base_rgb.ndim != 3 or base_rgb.shape[-1] != 3:
        raise ValueError("base_rgb must have shape (H, W, 3)")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    height, width, _ = base_rgb.shape
    alpha = soft_tapered_alpha_maps(
        p0,
        p1,
        p2,
        width_start,
        width_mid,
        width_end,
        opacities,
        image_size=(width, height),
        samples_per_curve=samples_per_curve,
        edge_softness_pixels=edge_softness_pixels,
    )
    return _ordered_composite(base_rgb, alpha, colors)


def render_soft_ellipses(
    centers: torch.Tensor,
    radii_x: torch.Tensor,
    radii_y: torch.Tensor,
    angles: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    base_rgb: torch.Tensor,
    edge_softness_pixels: float = 0.75,
) -> torch.Tensor:
    """Render oriented ellipse patches with pixel-scale soft hard edges."""
    if base_rgb.ndim != 3 or base_rgb.shape[-1] != 3:
        raise ValueError("base_rgb must have shape (H, W, 3)")
    height, width, _ = base_rgb.shape
    alpha = soft_ellipse_alpha_maps(
        centers,
        radii_x,
        radii_y,
        angles,
        opacities,
        image_size=(width, height),
        edge_softness_pixels=edge_softness_pixels,
    )
    return _ordered_composite(base_rgb, alpha, colors)
