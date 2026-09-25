"""Differentiable local refinement initialized from the residual painter."""

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
    """Summary of one local refinement run."""

    refined_strokes: int
    steps: int
    initial_loss: float
    final_loss: float
    device: str


def _stroke_center(stroke: Stroke) -> tuple[float, float]:
    return stroke.point_at(0.5)


def _select_refinement_indices(
    strokes: list[Stroke],
    target_rgb: np.ndarray,
    painted_rgb: np.ndarray,
    max_refine_strokes: int,
) -> list[int]:
    """Select strokes centered on the highest residual-error regions."""
    if max_refine_strokes < 1:
        raise ValueError("max_refine_strokes must be positive")

    residual = np.linalg.norm(target_rgb - painted_rgb, axis=2)
    height, width = residual.shape

    scored: list[tuple[float, int]] = []
    for index, stroke in enumerate(strokes):
        cx, cy = _stroke_center(stroke)
        x = min(width - 1, max(0, round(cx * (width - 1))))
        y = min(height - 1, max(0, round(cy * (height - 1))))
        scored.append((float(residual[y, x]), index))

    scored.sort(reverse=True)
    count = min(max_refine_strokes, len(strokes))
    return sorted(index for _, index in scored[:count])


def _resize_rgb(image_rgb: np.ndarray, resolution: int) -> np.ndarray:
    image = Image.fromarray(np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    image.thumbnail((resolution, resolution), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


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
) -> tuple[list[Stroke], tuple[int, int, int], RefinementStats]:
    """Refine appearance parameters while keeping residual stroke geometry fixed.

    The full residual stroke program is retained. Only strokes whose centers lie
    in the highest-error regions are selected, and only width, color, and opacity
    are optimized. Fixed geometry prevents long-range drift and stray-line
    artifacts while preserving the residual painter's spatial structure.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if optimization_resolution < 16:
        raise ValueError("optimization_resolution must be at least 16")

    strokes, background = paint_residual(
        image_rgb,
        palette,
        total_strokes,
        seed=seed,
    )

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

    fixed_base = render_strokes(
        fixed_strokes,
        size=(width, height),
        background=background,
    )
    base_rgb = np.asarray(fixed_base, dtype=np.float32) / 255.0

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    elif device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    else:
        resolved_device = device

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

    p0 = p0_init.clone()
    p1 = p1_init.clone()
    p2 = p2_init.clone()
    widths = torch.nn.Parameter(width_init.clone())
    colors = torch.nn.Parameter(color_init.clone())
    opacities = torch.nn.Parameter(opacity_init.clone())

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
        return torch.mean((rendered - target) ** 2)

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

    refined_selected: list[Stroke] = []
    for index in range(len(selected)):
        refined_selected.append(
            Stroke(
                p0=selected[index].p0,
                p1=selected[index].p1,
                p2=selected[index].p2,
                width=min(0.035, max(0.004, float(widths[index].detach().cpu().item()))),
                color=tuple(float(v) for v in colors[index].detach().cpu().tolist()),
                opacity=min(0.95, max(0.35, float(opacities[index].detach().cpu().item()))),
            )
        )

    refined_strokes = list(strokes)
    for stroke_index, refined in zip(selected_indices, refined_selected, strict=True):
        refined_strokes[stroke_index] = refined

    stats = RefinementStats(
        refined_strokes=len(selected_indices),
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        device=resolved_device,
    )
    return refined_strokes, background, stats
