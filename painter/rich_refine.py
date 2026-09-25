"""Ordered differentiable refinement for mixed rich primitive programs."""

from __future__ import annotations

import numpy as np
import torch

from painter.diffrender import (
    render_soft_ellipses,
    soft_tapered_alpha_maps,
)
from painter.refine import (
    RefinementStats,
    _objective_loss,
    _resize_rgb,
    _resolve_device,
    _validate_refinement_args,
)
from painter.renderer import render_primitive_overlay, render_strokes
from painter.rich import paint_region_rich_residual
from painter.stroke import EllipsePatch, Primitive, TaperedStroke


def _raster_segment_affine(
    primitives: list[TaperedStroke],
    *,
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize a fixed raster segment as canvas -> canvas * T + B."""
    width, height = size
    transmission = np.ones((height, width, 1), dtype=np.float32)
    bias = np.zeros((height, width, 3), dtype=np.float32)

    for primitive in primitives:
        overlay = render_primitive_overlay(
            primitive,
            size=size,
            samples_per_curve=32,
        )
        rgba = np.asarray(overlay, dtype=np.float32) / 255.0
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3]
        bias = bias * (1.0 - alpha) + rgb * alpha
        transmission = transmission * (1.0 - alpha)

    return (
        torch.tensor(transmission, dtype=dtype, device=device),
        torch.tensor(bias, dtype=dtype, device=device),
    )


def _apply_affine(
    canvas: torch.Tensor,
    transform: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    transmission, bias = transform
    return canvas * transmission + bias


def refine_region_rich_primitives_ordered(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_primitives: int,
    *,
    seed: int = 0,
    steps: int = 40,
    lr: float = 0.01,
    max_refine_strokes: int = 256,
    optimization_resolution: int = 96,
    device: str = "auto",
    objective: str = "mse",
    ssim_weight: float = 0.20,
    edge_weight: float = 0.10,
    geometry_bound: float = 0.03,
) -> tuple[list[Primitive], tuple[int, int, int], RefinementStats]:
    """Refine rich primitives while preserving the full ordered program context."""
    _validate_refinement_args(
        steps=steps,
        lr=lr,
        optimization_resolution=optimization_resolution,
        objective=objective,
        ssim_weight=ssim_weight,
        edge_weight=edge_weight,
    )
    if max_refine_strokes < 1:
        raise ValueError("max_refine_strokes must be positive")
    if not 0.0 <= geometry_bound <= 1.0:
        raise ValueError("geometry_bound must lie in [0, 1]")

    primitives, background = paint_region_rich_residual(
        image_rgb,
        palette,
        total_primitives,
        seed=seed,
    )
    patches = [primitive for primitive in primitives if isinstance(primitive, EllipsePatch)]
    tapered = [primitive for primitive in primitives if isinstance(primitive, TaperedStroke)]

    resolved_device = _resolve_device(device)
    torch_device = torch.device(resolved_device)
    dtype = torch.float32

    target_small = _resize_rgb(image_rgb, optimization_resolution)
    target = torch.tensor(target_small, dtype=dtype, device=torch_device)
    height, width, _ = target_small.shape
    size = (width, height)

    base_rgb = np.empty_like(target_small, dtype=np.float32)
    base_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    background_tensor = torch.tensor(base_rgb, dtype=dtype, device=torch_device)

    first_loss: float | None = None
    final_loss = 0.0
    refined_count = 0
    refined_patches = list(patches)

    # Patches occur before all tapered strokes in the rich painter. Preserve the
    # later stroke suffix during patch optimization instead of asking patches to
    # explain the complete target by themselves.
    if patches:
        suffix_transform = _raster_segment_affine(
            tapered,
            size=size,
            device=torch_device,
            dtype=dtype,
        )

        centers_init = torch.tensor(
            [patch.center for patch in patches],
            dtype=dtype,
            device=torch_device,
        )
        center_delta = torch.nn.Parameter(torch.zeros_like(centers_init))
        radii_x = torch.nn.Parameter(
            torch.tensor([patch.radius_x for patch in patches], dtype=dtype, device=torch_device)
        )
        radii_y = torch.nn.Parameter(
            torch.tensor([patch.radius_y for patch in patches], dtype=dtype, device=torch_device)
        )
        angles = torch.nn.Parameter(
            torch.tensor([patch.angle for patch in patches], dtype=dtype, device=torch_device)
        )
        colors = torch.nn.Parameter(
            torch.tensor([patch.color for patch in patches], dtype=dtype, device=torch_device)
        )
        opacities = torch.nn.Parameter(
            torch.tensor([patch.opacity for patch in patches], dtype=dtype, device=torch_device)
        )
        optimizer = torch.optim.Adam(
            [center_delta, radii_x, radii_y, angles, colors, opacities],
            lr=lr,
        )

        def patch_geometry() -> torch.Tensor:
            return torch.clamp(
                centers_init + geometry_bound * torch.tanh(center_delta),
                0.0,
                1.0,
            )

        def patch_loss() -> torch.Tensor:
            canvas = render_soft_ellipses(
                patch_geometry(),
                radii_x,
                radii_y,
                angles,
                colors,
                opacities,
                base_rgb=background_tensor,
            )
            canvas = _apply_affine(canvas, suffix_transform)
            return _objective_loss(
                canvas,
                target,
                objective=objective,
                ssim_weight=ssim_weight,
                edge_weight=edge_weight,
            )

        with torch.no_grad():
            first_loss = float(patch_loss().item())

        for _ in range(steps):
            optimizer.zero_grad()
            loss = patch_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                radii_x.clamp_(0.002, 0.20)
                radii_y.clamp_(0.002, 0.20)
                colors.clamp_(0.0, 1.0)
                opacities.clamp_(0.20, 1.0)

        with torch.no_grad():
            final_loss = float(patch_loss().item())
            final_centers = patch_geometry()

        refined_patches = [
            EllipsePatch(
                center=tuple(float(v) for v in final_centers[index].cpu().tolist()),
                radius_x=float(radii_x[index].cpu().item()),
                radius_y=float(radii_y[index].cpu().item()),
                angle=float(angles[index].cpu().item()),
                color=tuple(float(v) for v in colors[index].cpu().tolist()),
                opacity=float(opacities[index].cpu().item()),
            )
            for index in range(len(patches))
        ]
        refined_count += len(refined_patches)

    refined_tapered = list(tapered)
    if tapered:
        current_full = render_strokes(
            [*refined_patches, *tapered],
            size=(image_rgb.shape[1], image_rgb.shape[0]),
            background=background,
        )
        current_rgb = np.asarray(current_full, dtype=np.float32) / 255.0
        residual = np.linalg.norm(image_rgb - current_rgb, axis=2)
        full_height, full_width = residual.shape

        scored: list[tuple[float, int]] = []
        for index, stroke in enumerate(tapered):
            cx, cy = stroke.point_at(0.5)
            x = min(full_width - 1, max(0, round(cx * (full_width - 1))))
            y = min(full_height - 1, max(0, round(cy * (full_height - 1))))
            scored.append((float(residual[y, x]), index))
        scored.sort(reverse=True)
        selected_indices = sorted(
            index for _, index in scored[: min(max_refine_strokes, len(tapered))]
        )
        selected = [tapered[index] for index in selected_indices]

        base_image = render_strokes(
            refined_patches,
            size=size,
            background=background,
        )
        base = torch.tensor(
            np.asarray(base_image, dtype=np.float32) / 255.0,
            dtype=dtype,
            device=torch_device,
        )

        segments: list[list[TaperedStroke]] = []
        previous = 0
        for selected_index in selected_indices:
            segments.append(tapered[previous:selected_index])
            previous = selected_index + 1
        segments.append(tapered[previous:])
        segment_transforms = [
            _raster_segment_affine(
                segment,
                size=size,
                device=torch_device,
                dtype=dtype,
            )
            for segment in segments
        ]

        p0_init = torch.tensor([stroke.p0 for stroke in selected], dtype=dtype, device=torch_device)
        p1_init = torch.tensor([stroke.p1 for stroke in selected], dtype=dtype, device=torch_device)
        p2_init = torch.tensor([stroke.p2 for stroke in selected], dtype=dtype, device=torch_device)
        delta_p0 = torch.nn.Parameter(torch.zeros_like(p0_init))
        delta_p1 = torch.nn.Parameter(torch.zeros_like(p1_init))
        delta_p2 = torch.nn.Parameter(torch.zeros_like(p2_init))
        width_start = torch.nn.Parameter(
            torch.tensor(
                [stroke.width_start for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
        )
        width_mid = torch.nn.Parameter(
            torch.tensor(
                [stroke.width_mid for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
        )
        width_end = torch.nn.Parameter(
            torch.tensor(
                [stroke.width_end for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
        )
        colors = torch.nn.Parameter(
            torch.tensor([stroke.color for stroke in selected], dtype=dtype, device=torch_device)
        )
        opacities = torch.nn.Parameter(
            torch.tensor([stroke.opacity for stroke in selected], dtype=dtype, device=torch_device)
        )
        optimizer = torch.optim.Adam(
            [
                delta_p0,
                delta_p1,
                delta_p2,
                width_start,
                width_mid,
                width_end,
                colors,
                opacities,
            ],
            lr=lr,
        )

        def stroke_geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.clamp(p0_init + geometry_bound * torch.tanh(delta_p0), 0.0, 1.0),
                torch.clamp(p1_init + geometry_bound * torch.tanh(delta_p1), 0.0, 1.0),
                torch.clamp(p2_init + geometry_bound * torch.tanh(delta_p2), 0.0, 1.0),
            )

        def stroke_loss() -> torch.Tensor:
            p0, p1, p2 = stroke_geometry()
            alpha_maps = soft_tapered_alpha_maps(
                p0,
                p1,
                p2,
                width_start,
                width_mid,
                width_end,
                opacities,
                image_size=size,
            )

            canvas = base
            for local_index in range(len(selected_indices)):
                canvas = _apply_affine(canvas, segment_transforms[local_index])
                alpha = alpha_maps[local_index][..., None]
                color = colors[local_index][None, None, :]
                canvas = canvas * (1.0 - alpha) + color * alpha
            canvas = _apply_affine(canvas, segment_transforms[-1])

            return _objective_loss(
                canvas,
                target,
                objective=objective,
                ssim_weight=ssim_weight,
                edge_weight=edge_weight,
            )

        if first_loss is None:
            with torch.no_grad():
                first_loss = float(stroke_loss().item())

        for _ in range(steps):
            optimizer.zero_grad()
            loss = stroke_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                width_start.clamp_(0.001, 0.08)
                width_mid.clamp_(0.001, 0.08)
                width_end.clamp_(0.001, 0.08)
                colors.clamp_(0.0, 1.0)
                opacities.clamp_(0.20, 1.0)

        with torch.no_grad():
            final_loss = float(stroke_loss().item())
            final_p0, final_p1, final_p2 = stroke_geometry()

        for local_index, stroke_index in enumerate(selected_indices):
            refined_tapered[stroke_index] = TaperedStroke(
                p0=tuple(float(v) for v in final_p0[local_index].cpu().tolist()),
                p1=tuple(float(v) for v in final_p1[local_index].cpu().tolist()),
                p2=tuple(float(v) for v in final_p2[local_index].cpu().tolist()),
                width_start=float(width_start[local_index].cpu().item()),
                width_mid=float(width_mid[local_index].cpu().item()),
                width_end=float(width_end[local_index].cpu().item()),
                color=tuple(float(v) for v in colors[local_index].cpu().tolist()),
                opacity=float(opacities[local_index].cpu().item()),
            )
        refined_count += len(selected_indices)

    stats = RefinementStats(
        refined_strokes=refined_count,
        steps=steps * (2 if patches and tapered else 1),
        initial_loss=0.0 if first_loss is None else first_loss,
        final_loss=final_loss,
        device=resolved_device,
        objective=objective,
        optimize_geometry=True,
        geometry_bound=geometry_bound,
        continuous_color=True,
    )
    return [*refined_patches, *refined_tapered], background, stats
