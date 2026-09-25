"""Calibration helpers for real-vs-differentiable rich primitive rendering."""

from __future__ import annotations

import numpy as np
import torch

from painter.diffrender import render_soft_ellipses, render_soft_tapered_strokes
from painter.metrics import reconstruction_metrics
from painter.renderer import render_strokes
from painter.stroke import EllipsePatch, TaperedStroke


def rich_renderer_consistency(
    *,
    size: tuple[int, int] = (96, 96),
) -> dict[str, float]:
    """Measure mismatch between the Pillow renderer and soft rich rasterizer."""
    width, height = size
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

    dtype = torch.float32
    base = torch.empty((height, width, 3), dtype=dtype)
    base[...] = torch.tensor(background, dtype=dtype) / 255.0

    soft = render_soft_ellipses(
        torch.tensor([patch.center for patch in patches], dtype=dtype),
        torch.tensor([patch.radius_x for patch in patches], dtype=dtype),
        torch.tensor([patch.radius_y for patch in patches], dtype=dtype),
        torch.tensor([patch.angle for patch in patches], dtype=dtype),
        torch.tensor([patch.color for patch in patches], dtype=dtype),
        torch.tensor([patch.opacity for patch in patches], dtype=dtype),
        base_rgb=base,
    )
    soft = render_soft_tapered_strokes(
        torch.tensor([stroke.p0 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.p1 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.p2 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_start for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_mid for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_end for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.color for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.opacity for stroke in strokes], dtype=dtype),
        base_rgb=soft,
        samples_per_curve=32,
    )
    soft_rgb = soft.detach().cpu().numpy()
    return reconstruction_metrics(real_rgb, soft_rgb)



def dense_rich_renderer_consistency(
    *,
    size: tuple[int, int] = (72, 72),
    seed: int = 17,
) -> dict[str, float]:
    """Stress-test accumulated mismatch on a dense overlapping rich program."""
    width, height = size
    rng = np.random.default_rng(seed)
    background = (9, 13, 20)

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
        delta0 = rng.uniform(-0.22, 0.22, size=2)
        delta2 = rng.uniform(-0.22, 0.22, size=2)
        control = center + rng.uniform(-0.15, 0.15, size=2)
        p0 = np.clip(center + delta0, 0.02, 0.98)
        p1 = np.clip(control, 0.02, 0.98)
        p2 = np.clip(center + delta2, 0.02, 0.98)
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

    real = render_strokes(
        [*patches, *strokes],
        size=size,
        background=background,
        samples_per_curve=32,
    )
    real_rgb = np.asarray(real, dtype=np.float32) / 255.0

    dtype = torch.float32
    base = torch.empty((height, width, 3), dtype=dtype)
    base[...] = torch.tensor(background, dtype=dtype) / 255.0

    soft = render_soft_ellipses(
        torch.tensor([patch.center for patch in patches], dtype=dtype),
        torch.tensor([patch.radius_x for patch in patches], dtype=dtype),
        torch.tensor([patch.radius_y for patch in patches], dtype=dtype),
        torch.tensor([patch.angle for patch in patches], dtype=dtype),
        torch.tensor([patch.color for patch in patches], dtype=dtype),
        torch.tensor([patch.opacity for patch in patches], dtype=dtype),
        base_rgb=base,
    )
    soft = render_soft_tapered_strokes(
        torch.tensor([stroke.p0 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.p1 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.p2 for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_start for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_mid for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.width_end for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.color for stroke in strokes], dtype=dtype),
        torch.tensor([stroke.opacity for stroke in strokes], dtype=dtype),
        base_rgb=soft,
        samples_per_curve=32,
    )
    return reconstruction_metrics(real_rgb, soft.detach().cpu().numpy())
