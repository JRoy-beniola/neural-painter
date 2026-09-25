"""Calibration helpers for real-vs-differentiable rich primitive rendering."""

from __future__ import annotations

import numpy as np
import torch

from painter.diffrender import (
    render_soft_ellipses,
    render_soft_tapered_strokes,
    soft_tapered_alpha_maps,
)
from painter.metrics import reconstruction_metrics
from painter.renderer import render_primitive_overlay, render_strokes
from painter.stroke import EllipsePatch, TaperedStroke


def _background_tensor(
    background: tuple[int, int, int],
    *,
    size: tuple[int, int],
) -> torch.Tensor:
    width, height = size
    base = torch.empty((height, width, 3), dtype=torch.float32)
    base[...] = torch.tensor(background, dtype=torch.float32) / 255.0
    return base


def _raster_segment_affine(
    strokes: list[TaperedStroke],
    *,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = size
    transmission = np.ones((height, width, 1), dtype=np.float32)
    bias = np.zeros((height, width, 3), dtype=np.float32)
    for stroke in strokes:
        rgba = np.asarray(
            render_primitive_overlay(stroke, size=size, samples_per_curve=32),
            dtype=np.float32,
        ) / 255.0
        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3]
        bias = bias * (1.0 - alpha) + rgb * alpha
        transmission = transmission * (1.0 - alpha)
    return torch.tensor(transmission), torch.tensor(bias)


def _dense_program(
    *,
    seed: int,
) -> tuple[list[EllipsePatch], list[TaperedStroke]]:
    rng = np.random.default_rng(seed)
    patches: list[EllipsePatch] = []
    for _ in range(12):
        patches.append(
            EllipsePatch(
                center=(float(rng.uniform(0.22, 0.78)), float(rng.uniform(0.22, 0.78))),
                radius_x=float(rng.uniform(0.035, 0.14)),
                radius_y=float(rng.uniform(0.025, 0.10)),
                angle=float(rng.uniform(-1.2, 1.2)),
                color=tuple(float(v) for v in rng.uniform(0.08, 0.95, size=3)),
                opacity=float(rng.uniform(0.45, 0.88)),
            )
        )

    strokes: list[TaperedStroke] = []
    for _ in range(24):
        center = rng.uniform(0.25, 0.75, size=2)
        p0 = np.clip(center + rng.uniform(-0.22, 0.22, size=2), 0.02, 0.98)
        p1 = np.clip(center + rng.uniform(-0.15, 0.15, size=2), 0.02, 0.98)
        p2 = np.clip(center + rng.uniform(-0.22, 0.22, size=2), 0.02, 0.98)
        strokes.append(
            TaperedStroke(
                p0=(float(p0[0]), float(p0[1])),
                p1=(float(p1[0]), float(p1[1])),
                p2=(float(p2[0]), float(p2[1])),
                width_start=float(rng.uniform(0.008, 0.045)),
                width_mid=float(rng.uniform(0.02, 0.075)),
                width_end=float(rng.uniform(0.008, 0.045)),
                color=tuple(float(v) for v in rng.uniform(0.08, 0.98, size=3)),
                opacity=float(rng.uniform(0.45, 0.9)),
            )
        )
    return patches, strokes


def _soft_render(
    patches: list[EllipsePatch],
    strokes: list[TaperedStroke],
    *,
    size: tuple[int, int],
    background: tuple[int, int, int],
) -> np.ndarray:
    base = _background_tensor(background, size=size)
    soft = render_soft_ellipses(
        torch.tensor([patch.center for patch in patches], dtype=torch.float32),
        torch.tensor([patch.radius_x for patch in patches], dtype=torch.float32),
        torch.tensor([patch.radius_y for patch in patches], dtype=torch.float32),
        torch.tensor([patch.angle for patch in patches], dtype=torch.float32),
        torch.tensor([patch.color for patch in patches], dtype=torch.float32),
        torch.tensor([patch.opacity for patch in patches], dtype=torch.float32),
        base_rgb=base,
    )
    soft = render_soft_tapered_strokes(
        torch.tensor([stroke.p0 for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.p1 for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.p2 for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.width_start for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.width_mid for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.width_end for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.color for stroke in strokes], dtype=torch.float32),
        torch.tensor([stroke.opacity for stroke in strokes], dtype=torch.float32),
        base_rgb=soft,
        samples_per_curve=32,
    )
    return soft.detach().cpu().numpy()


def rich_renderer_consistency(
    *,
    size: tuple[int, int] = (96, 96),
) -> dict[str, float]:
    """Measure mismatch on a small mixed rich primitive program."""
    background = (12, 18, 24)
    patches = [
        EllipsePatch(
            center=(0.38, 0.42),
            radius_x=0.17,
            radius_y=0.09,
            angle=0.35,
            color=(0.2, 0.65, 0.85),
            opacity=0.78,
        ),
        EllipsePatch(
            center=(0.62, 0.58),
            radius_x=0.13,
            radius_y=0.07,
            angle=-0.55,
            color=(0.85, 0.55, 0.25),
            opacity=0.66,
        ),
    ]
    strokes = [
        TaperedStroke(
            p0=(0.15, 0.72),
            p1=(0.48, 0.22),
            p2=(0.84, 0.68),
            width_start=0.018,
            width_mid=0.075,
            width_end=0.026,
            color=(0.92, 0.95, 1.0),
            opacity=0.82,
        ),
        TaperedStroke(
            p0=(0.18, 0.28),
            p1=(0.55, 0.78),
            p2=(0.83, 0.32),
            width_start=0.032,
            width_mid=0.052,
            width_end=0.014,
            color=(0.15, 0.30, 0.72),
            opacity=0.72,
        ),
    ]
    real = render_strokes(
        [*patches, *strokes],
        size=size,
        background=background,
        samples_per_curve=32,
    )
    real_rgb = np.asarray(real, dtype=np.float32) / 255.0
    return reconstruction_metrics(
        real_rgb,
        _soft_render(patches, strokes, size=size, background=background),
    )


def dense_rich_renderer_consistency(
    *,
    size: tuple[int, int] = (72, 72),
    seed: int = 17,
) -> dict[str, float]:
    """Report full-soft accumulated mismatch on dense overlapping primitives."""
    background = (9, 13, 20)
    patches, strokes = _dense_program(seed=seed)
    real = render_strokes(
        [*patches, *strokes],
        size=size,
        background=background,
        samples_per_curve=32,
    )
    real_rgb = np.asarray(real, dtype=np.float32) / 255.0
    return reconstruction_metrics(
        real_rgb,
        _soft_render(patches, strokes, size=size, background=background),
    )


def ordered_context_renderer_consistency(
    *,
    size: tuple[int, int] = (72, 72),
    seed: int = 17,
    selected_count: int = 6,
) -> dict[str, dict[str, float]]:
    """Calibrate the two ordered-context forward passes used by refinement."""
    background = (9, 13, 20)
    patches, strokes = _dense_program(seed=seed)
    real = render_strokes(
        [*patches, *strokes],
        size=size,
        background=background,
        samples_per_curve=32,
    )
    real_rgb = np.asarray(real, dtype=np.float32) / 255.0

    # Patch-stage path: soft trainable patches, exact fixed raster stroke suffix.
    soft_patches = render_soft_ellipses(
        torch.tensor([patch.center for patch in patches], dtype=torch.float32),
        torch.tensor([patch.radius_x for patch in patches], dtype=torch.float32),
        torch.tensor([patch.radius_y for patch in patches], dtype=torch.float32),
        torch.tensor([patch.angle for patch in patches], dtype=torch.float32),
        torch.tensor([patch.color for patch in patches], dtype=torch.float32),
        torch.tensor([patch.opacity for patch in patches], dtype=torch.float32),
        base_rgb=_background_tensor(background, size=size),
    )
    transmission, bias = _raster_segment_affine(strokes, size=size)
    patch_context = soft_patches * transmission + bias

    # Stroke-stage path: exact raster patches and fixed stroke segments, only a
    # sparse selected subset uses differentiable coverage.
    base_image = render_strokes(patches, size=size, background=background)
    canvas = torch.tensor(np.asarray(base_image, dtype=np.float32) / 255.0)
    selected_indices = sorted(
        set(np.linspace(0, len(strokes) - 1, selected_count, dtype=int).tolist())
    )
    selected = [strokes[index] for index in selected_indices]

    segments: list[list[TaperedStroke]] = []
    previous = 0
    for selected_index in selected_indices:
        segments.append(strokes[previous:selected_index])
        previous = selected_index + 1
    segments.append(strokes[previous:])
    transforms = [_raster_segment_affine(segment, size=size) for segment in segments]

    alpha_maps = soft_tapered_alpha_maps(
        torch.tensor([stroke.p0 for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.p1 for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.p2 for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.width_start for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.width_mid for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.width_end for stroke in selected], dtype=torch.float32),
        torch.tensor([stroke.opacity for stroke in selected], dtype=torch.float32),
        image_size=size,
        samples_per_curve=32,
    )
    colors = torch.round(
        torch.tensor([stroke.color for stroke in selected], dtype=torch.float32) * 255.0
    ) / 255.0
    for local_index in range(len(selected)):
        transmission, bias = transforms[local_index]
        canvas = canvas * transmission + bias
        alpha = alpha_maps[local_index][..., None]
        canvas = canvas * (1.0 - alpha) + colors[local_index][None, None, :] * alpha
    transmission, bias = transforms[-1]
    stroke_context = canvas * transmission + bias

    return {
        "patch_stage": reconstruction_metrics(real_rgb, patch_context.detach().numpy()),
        "stroke_stage": reconstruction_metrics(real_rgb, stroke_context.detach().numpy()),
    }
