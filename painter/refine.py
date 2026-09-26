"""Differentiable refinement initialized from the residual painter."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from painter.diffrender import (
    render_soft_ellipses,
    render_soft_strokes,
    render_soft_tapered_strokes,
)
from painter.iterative import paint_residual
from painter.renderer import render_strokes
from painter.rich import paint_region_rich_residual
from painter.stroke import EllipsePatch, Primitive, Stroke, TaperedStroke


@dataclass(frozen=True, slots=True)
class RefinementStats:
    """Summary of one refinement run."""

    refined_strokes: int
    steps: int
    initial_loss: float
    final_loss: float
    device: str
    objective: str
    optimize_geometry: bool = False
    geometry_bound: float = 0.0
    continuous_color: bool = True
    stages: int = 1
    sweeps: int = 1
    accepted_by_raster: bool | None = None
    raster_mse_before: float | None = None
    raster_mse_after: float | None = None


def _stroke_center(stroke: Stroke) -> tuple[float, float]:
    return stroke.point_at(0.5)


def _select_refinement_indices(
    strokes: list[Stroke],
    target_rgb: np.ndarray,
    painted_rgb: np.ndarray,
    max_refine_strokes: int | None,
) -> list[int]:
    """Select strokes centered on the highest residual-error regions."""
    if max_refine_strokes is not None and max_refine_strokes < 1:
        raise ValueError("max_refine_strokes must be positive or None")
    if max_refine_strokes is None or max_refine_strokes >= len(strokes):
        return list(range(len(strokes)))

    residual = np.linalg.norm(target_rgb - painted_rgb, axis=2)
    height, width = residual.shape

    scored: list[tuple[float, int]] = []
    for index, stroke in enumerate(strokes):
        cx, cy = _stroke_center(stroke)
        x = min(width - 1, max(0, round(cx * (width - 1))))
        y = min(height - 1, max(0, round(cy * (height - 1))))
        scored.append((float(residual[y, x]), index))

    scored.sort(reverse=True)
    return sorted(index for _, index in scored[:max_refine_strokes])


def _resize_rgb(image_rgb: np.ndarray, resolution: int) -> np.ndarray:
    image = Image.fromarray(np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    image.thumbnail((resolution, resolution), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _ssim_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM loss for HWC RGB tensors in [0, 1]."""
    x = rendered.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    kernel = 7
    padding = kernel // 2

    mu_x = torch.nn.functional.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = torch.nn.functional.avg_pool2d(y, kernel, stride=1, padding=padding)
    sigma_x = torch.nn.functional.avg_pool2d(x * x, kernel, 1, padding) - mu_x * mu_x
    sigma_y = torch.nn.functional.avg_pool2d(y * y, kernel, 1, padding) - mu_y * mu_y
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, kernel, 1, padding) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - torch.clamp(score.mean(), 0.0, 1.0)


def _multi_scale_ssim_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Average SSIM loss over native, half, and quarter resolutions."""
    losses: list[torch.Tensor] = []
    x = rendered
    y = target
    for _ in range(3):
        losses.append(_ssim_loss(x, y))
        if min(x.shape[0], x.shape[1]) < 16:
            break
        x_chw = x.permute(2, 0, 1).unsqueeze(0)
        y_chw = y.permute(2, 0, 1).unsqueeze(0)
        x = torch.nn.functional.avg_pool2d(x_chw, 2, stride=2).squeeze(0).permute(1, 2, 0)
        y = torch.nn.functional.avg_pool2d(y_chw, 2, stride=2).squeeze(0).permute(1, 2, 0)
    return torch.stack(losses).mean()


def _sobel_components(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = image.mean(dim=2)[None, None, :, :]
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=image.dtype,
        device=image.device,
    ).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    return gx, gy


def _edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match Sobel edge magnitude between rendered and target images."""
    gx_rendered, gy_rendered = _sobel_components(rendered)
    gx_target, gy_target = _sobel_components(target)
    magnitude_rendered = torch.sqrt(gx_rendered * gx_rendered + gy_rendered * gy_rendered + 1e-8)
    magnitude_target = torch.sqrt(gx_target * gx_target + gy_target * gy_target + 1e-8)
    return torch.mean(torch.abs(magnitude_rendered - magnitude_target))


def _edge_orientation_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align local edge direction, weighted toward strong target contours."""
    gx_rendered, gy_rendered = _sobel_components(rendered)
    gx_target, gy_target = _sobel_components(target)
    mag_rendered = torch.sqrt(gx_rendered * gx_rendered + gy_rendered * gy_rendered + 1e-8)
    mag_target = torch.sqrt(gx_target * gx_target + gy_target * gy_target + 1e-8)

    dot = gx_rendered * gx_target + gy_rendered * gy_target
    cosine = dot / (mag_rendered * mag_target + 1e-6)
    weight = mag_target / (mag_target.mean().detach() + 1e-6)
    weight = torch.clamp(weight, 0.0, 4.0)
    return torch.mean(weight * (1.0 - torch.clamp(cosine, -1.0, 1.0)))


def _laplacian_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match second-order local structure to discourage fuzzy edge halos."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=rendered.dtype,
        device=rendered.device,
    )[None, None, :, :]
    x = rendered.mean(dim=2)[None, None, :, :]
    y = target.mean(dim=2)[None, None, :, :]
    lx = torch.nn.functional.conv2d(x, kernel, padding=1)
    ly = torch.nn.functional.conv2d(y, kernel, padding=1)
    return torch.mean(torch.abs(lx - ly))


def _target_edge_distance(target: torch.Tensor) -> torch.Tensor:
    """Compute a fixed normalized Euclidean distance map to target contours."""
    with torch.no_grad():
        gx, gy = _sobel_components(target)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)[0, 0]
        threshold = torch.quantile(magnitude.flatten(), 0.82)
        edges = (magnitude >= threshold).detach().cpu().numpy().astype(np.uint8)
        inverse = (1 - edges).astype(np.uint8)
        distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
        scale = max(float(distance.max()), 1.0)
        distance = distance / scale
    return torch.tensor(distance, dtype=target.dtype, device=target.device)


def _positional_contour_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    target_distance: torch.Tensor,
) -> torch.Tensor:
    """Penalize displaced rendered edges and missing target contours."""
    gx_rendered, gy_rendered = _sobel_components(rendered)
    gx_target, gy_target = _sobel_components(target)
    rendered_mag = torch.sqrt(gx_rendered * gx_rendered + gy_rendered * gy_rendered + 1e-8)[0, 0]
    target_mag = torch.sqrt(gx_target * gx_target + gy_target * gy_target + 1e-8)[0, 0]

    rendered_norm = rendered_mag / (rendered_mag.mean().detach() + 1e-6)
    forward = torch.mean(rendered_norm * target_distance)

    target_threshold = torch.quantile(target_mag.flatten().detach(), 0.82)
    target_edges = (target_mag >= target_threshold).to(target.dtype)
    local_rendered = torch.nn.functional.max_pool2d(
        rendered_norm[None, None, :, :],
        kernel_size=5,
        stride=1,
        padding=2,
    )[0, 0]
    coverage = torch.sum(target_edges * torch.exp(-local_rendered)) / (
        torch.sum(target_edges) + 1e-6
    )

    target_energy = target_mag.mean().detach()
    clutter = torch.relu(rendered_mag.mean() - 1.15 * target_energy) / (
        target_energy + 1e-6
    )
    return forward + 0.35 * coverage + 0.05 * clutter


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return device


def _objective_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    objective: str,
    ssim_weight: float,
    edge_weight: float,
    target_edge_distance: torch.Tensor | None = None,
) -> torch.Tensor:
    mse = torch.mean((rendered - target) ** 2)
    if objective == "mse":
        return mse
    if objective == "structure":
        return mse + ssim_weight * _ssim_loss(rendered, target) + edge_weight * _edge_loss(
            rendered,
            target,
        )
    contour = (
        mse
        + ssim_weight * _multi_scale_ssim_loss(rendered, target)
        + edge_weight * _edge_loss(rendered, target)
        + 0.05 * _edge_orientation_loss(rendered, target)
        + 0.05 * _laplacian_loss(rendered, target)
    )
    if objective == "contour":
        return contour
    if target_edge_distance is None:
        target_edge_distance = _target_edge_distance(target)
    return contour + 0.12 * _positional_contour_loss(
        rendered,
        target,
        target_edge_distance,
    )


def _validate_refinement_args(
    *,
    steps: int,
    lr: float,
    optimization_resolution: int,
    objective: str,
    ssim_weight: float,
    edge_weight: float,
) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if optimization_resolution < 16:
        raise ValueError("optimization_resolution must be at least 16")
    if objective not in {"mse", "structure", "contour", "positional_contour"}:
        raise ValueError(
            "objective must be one of: mse, structure, contour, positional_contour"
        )
    if ssim_weight < 0.0 or edge_weight < 0.0:
        raise ValueError("structure loss weights must be non-negative")


def refine_residual_strokes(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_strokes: int,
    *,
    seed: int = 0,
    steps: int = 30,
    lr: float = 0.015,
    max_refine_strokes: int = 12,
    optimization_resolution: int = 64,
    device: str = "auto",
    objective: str = "mse",
    ssim_weight: float = 0.20,
    edge_weight: float = 0.10,
) -> tuple[list[Stroke], tuple[int, int, int], RefinementStats]:
    """Refine appearance parameters while keeping residual stroke geometry fixed."""
    _validate_refinement_args(
        steps=steps,
        lr=lr,
        optimization_resolution=optimization_resolution,
        objective=objective,
        ssim_weight=ssim_weight,
        edge_weight=edge_weight,
    )

    strokes, background = paint_residual(image_rgb, palette, total_strokes, seed=seed)
    full_initial = render_strokes(
        strokes,
        size=(image_rgb.shape[1], image_rgb.shape[0]),
        background=background,
    )
    full_initial_rgb = np.asarray(full_initial, dtype=np.float32) / 255.0

    selected_indices = _select_refinement_indices(
        strokes,
        image_rgb,
        full_initial_rgb,
        max_refine_strokes,
    )
    selected_set = set(selected_indices)
    fixed_strokes = [stroke for i, stroke in enumerate(strokes) if i not in selected_set]
    selected = [strokes[i] for i in selected_indices]

    target_small = _resize_rgb(image_rgb, optimization_resolution)
    height, width, _ = target_small.shape
    fixed_base = render_strokes(fixed_strokes, size=(width, height), background=background)
    base_rgb = np.asarray(fixed_base, dtype=np.float32) / 255.0

    resolved_device = _resolve_device(device)
    torch_device = torch.device(resolved_device)
    dtype = torch.float32

    p0 = torch.tensor([stroke.p0 for stroke in selected], dtype=dtype, device=torch_device)
    p1 = torch.tensor([stroke.p1 for stroke in selected], dtype=dtype, device=torch_device)
    p2 = torch.tensor([stroke.p2 for stroke in selected], dtype=dtype, device=torch_device)
    widths = torch.nn.Parameter(
        torch.tensor([stroke.width for stroke in selected], dtype=dtype, device=torch_device)
    )
    colors = torch.nn.Parameter(
        torch.tensor([stroke.color for stroke in selected], dtype=dtype, device=torch_device)
    )
    opacities = torch.nn.Parameter(
        torch.tensor([stroke.opacity for stroke in selected], dtype=dtype, device=torch_device)
    )

    optimizer = torch.optim.Adam([widths, colors, opacities], lr=lr)
    target = torch.tensor(target_small, dtype=dtype, device=torch_device)
    base = torch.tensor(base_rgb, dtype=dtype, device=torch_device)

    def loss_value() -> torch.Tensor:
        rendered = render_soft_strokes(
            p0,
            p1,
            p2,
            widths,
            colors,
            opacities,
            base_rgb=base,
        )
        return _objective_loss(
            rendered,
            target,
            objective=objective,
            ssim_weight=ssim_weight,
            edge_weight=edge_weight,
        )

    with torch.no_grad():
        initial_loss = float(loss_value().item())

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            widths.clamp_(0.004, 0.035)
            colors.clamp_(0.0, 1.0)
            opacities.clamp_(0.35, 0.95)

    with torch.no_grad():
        final_loss = float(loss_value().item())

    refined_strokes = list(strokes)
    for local_index, stroke_index in enumerate(selected_indices):
        source = selected[local_index]
        refined_strokes[stroke_index] = Stroke(
            p0=source.p0,
            p1=source.p1,
            p2=source.p2,
            width=min(0.035, max(0.004, float(widths[local_index].detach().cpu().item()))),
            color=tuple(float(v) for v in colors[local_index].detach().cpu().tolist()),
            opacity=min(
                0.95,
                max(0.35, float(opacities[local_index].detach().cpu().item())),
            ),
        )

    stats = RefinementStats(
        refined_strokes=len(selected_indices),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        device=resolved_device,
        objective=objective,
    )
    return refined_strokes, background, stats


def refine_residual_strokes_global(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_strokes: int,
    *,
    seed: int = 0,
    steps: int = 60,
    lr: float = 0.01,
    optimized_stroke_count: int | None = 256,
    optimization_resolution: int = 96,
    device: str = "auto",
    objective: str = "mse",
    ssim_weight: float = 0.20,
    edge_weight: float = 0.10,
    optimize_geometry: bool = True,
    geometry_bound: float = 0.03,
    continuous_color: bool = True,
) -> tuple[list[Stroke], tuple[int, int, int], RefinementStats]:
    """Audit reconstruction capacity with bounded optimization over many strokes.

    Geometry is parameterized as p = p_init + bound * tanh(delta), preventing
    the long-range drift seen in unconstrained refinement while allowing strokes
    to reposition locally. Width, opacity, and optionally continuous RGB color
    are optimized jointly.
    """
    _validate_refinement_args(
        steps=steps,
        lr=lr,
        optimization_resolution=optimization_resolution,
        objective=objective,
        ssim_weight=ssim_weight,
        edge_weight=edge_weight,
    )
    if optimized_stroke_count is not None and optimized_stroke_count < 1:
        raise ValueError("optimized_stroke_count must be positive or None")
    if geometry_bound < 0.0 or geometry_bound > 1.0:
        raise ValueError("geometry_bound must lie in [0, 1]")

    strokes, background = paint_residual(image_rgb, palette, total_strokes, seed=seed)
    full_initial = render_strokes(
        strokes,
        size=(image_rgb.shape[1], image_rgb.shape[0]),
        background=background,
    )
    full_initial_rgb = np.asarray(full_initial, dtype=np.float32) / 255.0

    selected_indices = _select_refinement_indices(
        strokes,
        image_rgb,
        full_initial_rgb,
        optimized_stroke_count,
    )
    selected_set = set(selected_indices)
    fixed_strokes = [stroke for i, stroke in enumerate(strokes) if i not in selected_set]
    selected = [strokes[i] for i in selected_indices]

    target_small = _resize_rgb(image_rgb, optimization_resolution)
    height, width, _ = target_small.shape
    fixed_base = render_strokes(fixed_strokes, size=(width, height), background=background)
    base_rgb = np.asarray(fixed_base, dtype=np.float32) / 255.0

    resolved_device = _resolve_device(device)
    torch_device = torch.device(resolved_device)
    dtype = torch.float32

    p0_init = torch.tensor([stroke.p0 for stroke in selected], dtype=dtype, device=torch_device)
    p1_init = torch.tensor([stroke.p1 for stroke in selected], dtype=dtype, device=torch_device)
    p2_init = torch.tensor([stroke.p2 for stroke in selected], dtype=dtype, device=torch_device)
    width_init = torch.tensor([stroke.width for stroke in selected], dtype=dtype, device=torch_device)
    color_init = torch.tensor([stroke.color for stroke in selected], dtype=dtype, device=torch_device)
    opacity_init = torch.tensor(
        [stroke.opacity for stroke in selected],
        dtype=dtype,
        device=torch_device,
    )

    delta_p0 = torch.nn.Parameter(torch.zeros_like(p0_init))
    delta_p1 = torch.nn.Parameter(torch.zeros_like(p1_init))
    delta_p2 = torch.nn.Parameter(torch.zeros_like(p2_init))
    widths = torch.nn.Parameter(width_init.clone())
    colors = torch.nn.Parameter(color_init.clone())
    opacities = torch.nn.Parameter(opacity_init.clone())

    parameters: list[torch.nn.Parameter] = [widths, opacities]
    if continuous_color:
        parameters.append(colors)
    if optimize_geometry and geometry_bound > 0.0:
        parameters.extend([delta_p0, delta_p1, delta_p2])

    optimizer = torch.optim.Adam(parameters, lr=lr)
    target = torch.tensor(target_small, dtype=dtype, device=torch_device)
    base = torch.tensor(base_rgb, dtype=dtype, device=torch_device)

    def geometry() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not optimize_geometry or geometry_bound == 0.0:
            return p0_init, p1_init, p2_init
        p0 = torch.clamp(p0_init + geometry_bound * torch.tanh(delta_p0), 0.0, 1.0)
        p1 = torch.clamp(p1_init + geometry_bound * torch.tanh(delta_p1), 0.0, 1.0)
        p2 = torch.clamp(p2_init + geometry_bound * torch.tanh(delta_p2), 0.0, 1.0)
        return p0, p1, p2

    def loss_value() -> torch.Tensor:
        p0, p1, p2 = geometry()
        rendered = render_soft_strokes(
            p0,
            p1,
            p2,
            widths,
            colors if continuous_color else color_init,
            opacities,
            base_rgb=base,
        )
        return _objective_loss(
            rendered,
            target,
            objective=objective,
            ssim_weight=ssim_weight,
            edge_weight=edge_weight,
        )

    with torch.no_grad():
        initial_loss = float(loss_value().item())

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            widths.clamp_(0.002, 0.06)
            if continuous_color:
                colors.clamp_(0.0, 1.0)
            opacities.clamp_(0.20, 1.0)

    with torch.no_grad():
        final_loss = float(loss_value().item())
        final_p0, final_p1, final_p2 = geometry()

    refined_strokes = list(strokes)
    for local_index, stroke_index in enumerate(selected_indices):
        source = selected[local_index]
        if optimize_geometry and geometry_bound > 0.0:
            p0 = tuple(float(v) for v in final_p0[local_index].cpu().tolist())
            p1 = tuple(float(v) for v in final_p1[local_index].cpu().tolist())
            p2 = tuple(float(v) for v in final_p2[local_index].cpu().tolist())
        else:
            p0, p1, p2 = source.p0, source.p1, source.p2

        color_tensor = colors if continuous_color else color_init
        refined_strokes[stroke_index] = Stroke(
            p0=p0,
            p1=p1,
            p2=p2,
            width=min(0.06, max(0.002, float(widths[local_index].cpu().item()))),
            color=tuple(float(v) for v in color_tensor[local_index].cpu().tolist()),
            opacity=min(1.0, max(0.20, float(opacities[local_index].cpu().item()))),
        )

    stats = RefinementStats(
        refined_strokes=len(selected_indices),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        device=resolved_device,
        objective=objective,
        optimize_geometry=optimize_geometry,
        geometry_bound=geometry_bound,
        continuous_color=continuous_color,
    )
    return refined_strokes, background, stats



def refine_residual_strokes_staged(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    total_strokes: int,
    *,
    seed: int = 0,
    stages: int = 4,
    sweeps: int = 1,
    batch_size: int = 256,
    steps_per_stage: int = 40,
    lr: float = 0.01,
    optimization_resolution: int = 96,
    device: str = "auto",
    objective: str = "mse",
    ssim_weight: float = 0.20,
    edge_weight: float = 0.10,
    optimize_geometry: bool = True,
    geometry_bound: float = 0.03,
    continuous_color: bool = True,
) -> tuple[list[Stroke], tuple[int, int, int], RefinementStats]:
    """Progressively refine disjoint high-error stroke batches.

    Each stage rerenders the current program, recomputes residual error, selects
    the highest-error strokes not optimized in an earlier stage, and performs a
    bounded differentiable update on that batch. This lets large stroke programs
    be repaired progressively without placing all strokes in one autograd graph.
    """
    _validate_refinement_args(
        steps=steps_per_stage,
        lr=lr,
        optimization_resolution=optimization_resolution,
        objective=objective,
        ssim_weight=ssim_weight,
        edge_weight=edge_weight,
    )
    if stages < 1:
        raise ValueError("stages must be positive")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if geometry_bound < 0.0 or geometry_bound > 1.0:
        raise ValueError("geometry_bound must lie in [0, 1]")

    strokes, background = paint_residual(image_rgb, palette, total_strokes, seed=seed)
    resolved_device = _resolve_device(device)
    torch_device = torch.device(resolved_device)
    dtype = torch.float32

    target_small = _resize_rgb(image_rgb, optimization_resolution)
    target = torch.tensor(target_small, dtype=dtype, device=torch_device)
    opt_height, opt_width, _ = target_small.shape

    all_refined_indices: set[int] = set()
    first_loss: float | None = None
    last_loss = 0.0
    completed_stages = 0
    completed_sweeps = 0

    for _sweep in range(sweeps):
        refined_this_sweep: set[int] = set()
        sweep_stages = 0

        for _stage in range(stages):
            if len(refined_this_sweep) >= len(strokes):
                break

            current_full = render_strokes(
                strokes,
                size=(image_rgb.shape[1], image_rgb.shape[0]),
                background=background,
            )
            current_rgb = np.asarray(current_full, dtype=np.float32) / 255.0
            residual = np.linalg.norm(image_rgb - current_rgb, axis=2)
            height, width = residual.shape

            scored: list[tuple[float, int]] = []
            for index, stroke in enumerate(strokes):
                if index in refined_this_sweep:
                    continue
                cx, cy = _stroke_center(stroke)
                x = min(width - 1, max(0, round(cx * (width - 1))))
                y = min(height - 1, max(0, round(cy * (height - 1))))
                scored.append((float(residual[y, x]), index))

            scored.sort(reverse=True)
            selected_indices = sorted(index for _, index in scored[:batch_size])
            if not selected_indices:
                break

            selected_set = set(selected_indices)
            fixed_strokes = [
                stroke for index, stroke in enumerate(strokes) if index not in selected_set
            ]
            selected = [strokes[index] for index in selected_indices]

            fixed_base = render_strokes(
                fixed_strokes,
                size=(opt_width, opt_height),
                background=background,
            )
            base_rgb = np.asarray(fixed_base, dtype=np.float32) / 255.0
            base = torch.tensor(base_rgb, dtype=dtype, device=torch_device)

            p0_init = torch.tensor(
                [stroke.p0 for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
            p1_init = torch.tensor(
                [stroke.p1 for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
            p2_init = torch.tensor(
                [stroke.p2 for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
            width_init = torch.tensor(
                [stroke.width for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
            color_init = torch.tensor(
                [stroke.color for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
            opacity_init = torch.tensor(
                [stroke.opacity for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )

            delta_p0 = torch.nn.Parameter(torch.zeros_like(p0_init))
            delta_p1 = torch.nn.Parameter(torch.zeros_like(p1_init))
            delta_p2 = torch.nn.Parameter(torch.zeros_like(p2_init))
            widths = torch.nn.Parameter(width_init.clone())
            colors = torch.nn.Parameter(color_init.clone())
            opacities = torch.nn.Parameter(opacity_init.clone())

            parameters: list[torch.nn.Parameter] = [widths, opacities]
            if continuous_color:
                parameters.append(colors)
            if optimize_geometry and geometry_bound > 0.0:
                parameters.extend([delta_p0, delta_p1, delta_p2])

            optimizer = torch.optim.Adam(parameters, lr=lr)

            def geometry(
                p0_init: torch.Tensor = p0_init,
                p1_init: torch.Tensor = p1_init,
                p2_init: torch.Tensor = p2_init,
                delta_p0: torch.Tensor = delta_p0,
                delta_p1: torch.Tensor = delta_p1,
                delta_p2: torch.Tensor = delta_p2,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if not optimize_geometry or geometry_bound == 0.0:
                    return p0_init, p1_init, p2_init
                p0 = torch.clamp(
                    p0_init + geometry_bound * torch.tanh(delta_p0),
                    0.0,
                    1.0,
                )
                p1 = torch.clamp(
                    p1_init + geometry_bound * torch.tanh(delta_p1),
                    0.0,
                    1.0,
                )
                p2 = torch.clamp(
                    p2_init + geometry_bound * torch.tanh(delta_p2),
                    0.0,
                    1.0,
                )
                return p0, p1, p2

            def loss_value(
                geometry=geometry,
                widths: torch.Tensor = widths,
                colors: torch.Tensor = colors,
                color_init: torch.Tensor = color_init,
                opacities: torch.Tensor = opacities,
                base: torch.Tensor = base,
            ) -> torch.Tensor:
                p0, p1, p2 = geometry()
                rendered = render_soft_strokes(
                    p0,
                    p1,
                    p2,
                    widths,
                    colors if continuous_color else color_init,
                    opacities,
                    base_rgb=base,
                )
                return _objective_loss(
                    rendered,
                    target,
                    objective=objective,
                    ssim_weight=ssim_weight,
                    edge_weight=edge_weight,
                )

            with torch.no_grad():
                stage_initial_loss = float(loss_value().item())
                if first_loss is None:
                    first_loss = stage_initial_loss

            for _ in range(steps_per_stage):
                optimizer.zero_grad()
                loss = loss_value()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    widths.clamp_(0.002, 0.06)
                    if continuous_color:
                        colors.clamp_(0.0, 1.0)
                    opacities.clamp_(0.20, 1.0)

            with torch.no_grad():
                last_loss = float(loss_value().item())
                final_p0, final_p1, final_p2 = geometry()

            color_tensor = colors if continuous_color else color_init
            for local_index, stroke_index in enumerate(selected_indices):
                source = selected[local_index]
                if optimize_geometry and geometry_bound > 0.0:
                    p0 = tuple(float(v) for v in final_p0[local_index].cpu().tolist())
                    p1 = tuple(float(v) for v in final_p1[local_index].cpu().tolist())
                    p2 = tuple(float(v) for v in final_p2[local_index].cpu().tolist())
                else:
                    p0, p1, p2 = source.p0, source.p1, source.p2

                strokes[stroke_index] = Stroke(
                    p0=p0,
                    p1=p1,
                    p2=p2,
                    width=min(
                        0.06,
                        max(0.002, float(widths[local_index].cpu().item())),
                    ),
                    color=tuple(
                        float(v) for v in color_tensor[local_index].cpu().tolist()
                    ),
                    opacity=min(
                        1.0,
                        max(0.20, float(opacities[local_index].cpu().item())),
                    ),
                )

            refined_this_sweep.update(selected_indices)
            all_refined_indices.update(selected_indices)
            completed_stages += 1
            sweep_stages += 1

        if sweep_stages == 0:
            break
        completed_sweeps += 1

    stats = RefinementStats(
        refined_strokes=len(all_refined_indices),
        steps=completed_stages * steps_per_stage,
        initial_loss=0.0 if first_loss is None else first_loss,
        final_loss=last_loss,
        device=resolved_device,
        objective=objective,
        optimize_geometry=optimize_geometry,
        geometry_bound=geometry_bound,
        continuous_color=continuous_color,
        stages=completed_stages,
        sweeps=completed_sweeps,
    )
    return strokes, background, stats



def refine_region_rich_primitives(
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
    """Refine fitted ellipse patches and high-error tapered detail strokes."""
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
    if geometry_bound < 0.0 or geometry_bound > 1.0:
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
    background_rgb = np.empty_like(target_small, dtype=np.float32)
    background_rgb[...] = np.asarray(background, dtype=np.float32) / 255.0
    base_background = torch.tensor(background_rgb, dtype=dtype, device=torch_device)

    first_loss: float | None = None
    last_loss = 0.0
    refined_count = 0

    refined_patches = list(patches)
    if patches:
        centers_init = torch.tensor(
            [patch.center for patch in patches],
            dtype=dtype,
            device=torch_device,
        )
        rx_init = torch.tensor(
            [patch.radius_x for patch in patches],
            dtype=dtype,
            device=torch_device,
        )
        ry_init = torch.tensor(
            [patch.radius_y for patch in patches],
            dtype=dtype,
            device=torch_device,
        )
        angle_init = torch.tensor(
            [patch.angle for patch in patches],
            dtype=dtype,
            device=torch_device,
        )
        center_delta = torch.nn.Parameter(torch.zeros_like(centers_init))
        rx = torch.nn.Parameter(rx_init.clone())
        ry = torch.nn.Parameter(ry_init.clone())
        angles = torch.nn.Parameter(angle_init.clone())
        colors = torch.nn.Parameter(
            torch.tensor(
                [patch.color for patch in patches],
                dtype=dtype,
                device=torch_device,
            )
        )
        opacities = torch.nn.Parameter(
            torch.tensor(
                [patch.opacity for patch in patches],
                dtype=dtype,
                device=torch_device,
            )
        )
        optimizer = torch.optim.Adam([center_delta, rx, ry, angles, colors, opacities], lr=lr)

        def patch_loss() -> torch.Tensor:
            centers = torch.clamp(
                centers_init + geometry_bound * torch.tanh(center_delta),
                0.0,
                1.0,
            )
            rendered = render_soft_ellipses(
                centers,
                rx,
                ry,
                angles,
                colors,
                opacities,
                base_rgb=base_background,
            )
            return _objective_loss(
                rendered,
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
                rx.clamp_(0.002, 0.20)
                ry.clamp_(0.002, 0.20)
                colors.clamp_(0.0, 1.0)
                opacities.clamp_(0.20, 1.0)

        with torch.no_grad():
            last_loss = float(patch_loss().item())
            final_centers = torch.clamp(
                centers_init + geometry_bound * torch.tanh(center_delta),
                0.0,
                1.0,
            )

        refined_patches = [
            EllipsePatch(
                center=tuple(float(v) for v in final_centers[index].cpu().tolist()),
                radius_x=float(rx[index].cpu().item()),
                radius_y=float(ry[index].cpu().item()),
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
        full_h, full_w = residual.shape
        scored: list[tuple[float, int]] = []
        for index, stroke in enumerate(tapered):
            cx, cy = stroke.point_at(0.5)
            x = min(full_w - 1, max(0, round(cx * (full_w - 1))))
            y = min(full_h - 1, max(0, round(cy * (full_h - 1))))
            scored.append((float(residual[y, x]), index))
        scored.sort(reverse=True)
        selected_indices = sorted(
            index for _, index in scored[: min(max_refine_strokes, len(tapered))]
        )
        selected_set = set(selected_indices)
        fixed_tapered = [
            stroke for index, stroke in enumerate(tapered) if index not in selected_set
        ]
        selected = [tapered[index] for index in selected_indices]

        fixed_canvas = render_strokes(
            [*refined_patches, *fixed_tapered],
            size=(width, height),
            background=background,
        )
        fixed_rgb = np.asarray(fixed_canvas, dtype=np.float32) / 255.0
        base = torch.tensor(fixed_rgb, dtype=dtype, device=torch_device)

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
            torch.tensor(
                [stroke.opacity for stroke in selected],
                dtype=dtype,
                device=torch_device,
            )
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
            rendered = render_soft_tapered_strokes(
                p0,
                p1,
                p2,
                width_start,
                width_mid,
                width_end,
                colors,
                opacities,
                base_rgb=base,
            )
            return _objective_loss(
                rendered,
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
            last_loss = float(stroke_loss().item())
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

    refined: list[Primitive] = [*refined_patches, *refined_tapered]
    stats = RefinementStats(
        refined_strokes=refined_count,
        steps=steps * (2 if patches and tapered else 1),
        initial_loss=0.0 if first_loss is None else first_loss,
        final_loss=last_loss,
        device=resolved_device,
        objective=objective,
        optimize_geometry=True,
        geometry_bound=geometry_bound,
        continuous_color=True,
    )
    return refined, background, stats
