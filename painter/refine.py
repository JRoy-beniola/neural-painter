"""Differentiable refinement initialized from the residual painter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from painter.diffrender import render_soft_strokes
from painter.iterative import paint_residual
from painter.renderer import render_strokes
from painter.stroke import Stroke


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


def _edge_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match Sobel edge magnitude between rendered and target images."""
    x = rendered.mean(dim=2)[None, None, :, :]
    y = target.mean(dim=2)[None, None, :, :]

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=rendered.dtype,
        device=rendered.device,
    ).unsqueeze(0)
    sobel_y = sobel_x.transpose(-1, -2)

    def magnitude(image: torch.Tensor) -> torch.Tensor:
        gx = torch.nn.functional.conv2d(image, sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(image, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    return torch.mean(torch.abs(magnitude(x) - magnitude(y)))


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
) -> torch.Tensor:
    mse = torch.mean((rendered - target) ** 2)
    if objective == "mse":
        return mse
    return mse + ssim_weight * _ssim_loss(rendered, target) + edge_weight * _edge_loss(
        rendered,
        target,
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
    if objective not in {"mse", "structure"}:
        raise ValueError("objective must be one of: mse, structure")
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
