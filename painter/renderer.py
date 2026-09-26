"""Deterministic raster rendering for Phase 0 stroke programs."""

from __future__ import annotations

import math
from collections.abc import Iterable

from PIL import Image, ImageDraw

from painter.stroke import BezierRibbon, EllipsePatch, PolygonPatch, Primitive, Stroke, TaperedStroke


def _to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x = point[0] * (width - 1)
    y = point[1] * (height - 1)
    return x, y


def _rgb8(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in color)


def _render_constant_stroke(
    draw: ImageDraw.ImageDraw,
    stroke: Stroke,
    *,
    width: int,
    height: int,
    samples_per_curve: int,
) -> None:
    points = [
        _to_pixel(stroke.point_at(i / (samples_per_curve - 1)), width, height)
        for i in range(samples_per_curve)
    ]
    pixel_width = max(1, round(stroke.width * min(width, height)))
    alpha = round(stroke.opacity * 255)
    draw.line(
        points,
        fill=(*_rgb8(stroke.color), alpha),
        width=pixel_width,
        joint="curve",
    )


def _render_tapered_stroke(
    draw: ImageDraw.ImageDraw,
    stroke: TaperedStroke,
    *,
    width: int,
    height: int,
    samples_per_curve: int,
) -> None:
    alpha = round(stroke.opacity * 255)
    fill = (*_rgb8(stroke.color), alpha)
    points = [
        _to_pixel(stroke.point_at(i / (samples_per_curve - 1)), width, height)
        for i in range(samples_per_curve)
    ]
    scale = min(width, height)

    for index in range(samples_per_curve - 1):
        t_mid = (index + 0.5) / (samples_per_curve - 1)
        pixel_width = max(1, round(stroke.width_at(t_mid) * scale))
        draw.line(
            [points[index], points[index + 1]],
            fill=fill,
            width=pixel_width,
        )


def _render_patch(
    draw: ImageDraw.ImageDraw,
    patch: EllipsePatch,
    *,
    width: int,
    height: int,
    samples: int = 40,
) -> None:
    cx, cy = _to_pixel(patch.center, width, height)
    rx = patch.radius_x * (width - 1)
    ry = patch.radius_y * (height - 1)
    cos_a = math.cos(patch.angle)
    sin_a = math.sin(patch.angle)

    points: list[tuple[float, float]] = []
    for index in range(samples):
        theta = 2.0 * math.pi * index / samples
        ex = rx * math.cos(theta)
        ey = ry * math.sin(theta)
        points.append(
            (
                cx + ex * cos_a - ey * sin_a,
                cy + ex * sin_a + ey * cos_a,
            )
        )

    alpha = round(patch.opacity * 255)
    draw.polygon(points, fill=(*_rgb8(patch.color), alpha))


def _render_polygon_patch(
    draw: ImageDraw.ImageDraw,
    patch: PolygonPatch,
    *,
    width: int,
    height: int,
) -> None:
    points = [_to_pixel(point, width, height) for point in patch.vertices]
    alpha = round(patch.opacity * 255)
    draw.polygon(points, fill=(*_rgb8(patch.color), alpha))


def _render_bezier_ribbon(
    draw: ImageDraw.ImageDraw,
    ribbon: BezierRibbon,
    *,
    width: int,
    height: int,
    samples_per_curve: int,
) -> None:
    scale = min(width, height)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []

    for index in range(samples_per_curve):
        t = index / (samples_per_curve - 1)
        cx, cy = _to_pixel(ribbon.point_at(t), width, height)
        tx, ty = ribbon.tangent_at(t)
        norm = math.hypot(tx * (width - 1), ty * (height - 1))
        if norm < 1e-8:
            nx, ny = 0.0, 1.0
        else:
            nx = -(ty * (height - 1)) / norm
            ny = (tx * (width - 1)) / norm
        half_width = 0.5 * ribbon.width_at(t) * scale
        left.append((cx + nx * half_width, cy + ny * half_width))
        right.append((cx - nx * half_width, cy - ny * half_width))

    polygon = [*left, *reversed(right)]
    alpha = round(ribbon.opacity * 255)
    draw.polygon(polygon, fill=(*_rgb8(ribbon.color), alpha))


def render_primitive_overlay(
    primitive: Primitive,
    *,
    size: tuple[int, int],
    samples_per_curve: int = 32,
) -> Image.Image:
    """Rasterize one primitive to an RGBA overlay without a background."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if isinstance(primitive, Stroke):
        _render_constant_stroke(
            draw,
            primitive,
            width=width,
            height=height,
            samples_per_curve=samples_per_curve,
        )
    elif isinstance(primitive, TaperedStroke):
        _render_tapered_stroke(
            draw,
            primitive,
            width=width,
            height=height,
            samples_per_curve=samples_per_curve,
        )
    elif isinstance(primitive, EllipsePatch):
        _render_patch(draw, primitive, width=width, height=height)
    elif isinstance(primitive, PolygonPatch):
        _render_polygon_patch(draw, primitive, width=width, height=height)
    elif isinstance(primitive, BezierRibbon):
        _render_bezier_ribbon(
            draw,
            primitive,
            width=width,
            height=height,
            samples_per_curve=samples_per_curve,
        )
    else:
        raise TypeError(f"unsupported primitive type: {type(primitive)!r}")

    return overlay


def render_strokes(
    strokes: Iterable[Primitive],
    *,
    size: tuple[int, int],
    background: tuple[int, int, int] = (255, 255, 255),
    samples_per_curve: int = 32,
) -> Image.Image:
    """Render an ordered primitive program to an RGB Pillow image.

    Existing constant-width strokes, tapered Bezier strokes, and oriented ellipse
    patches share the same ordered compositing program. Later primitives are
    composited over earlier ones.
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    canvas = Image.new("RGB", size, background)

    for primitive in strokes:
        overlay = render_primitive_overlay(
            primitive,
            size=size,
            samples_per_curve=samples_per_curve,
        )
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    return canvas
