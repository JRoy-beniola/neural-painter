"""Deterministic raster rendering for Phase 0 stroke programs."""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageDraw

from painter.stroke import Stroke


def _to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x = point[0] * (width - 1)
    y = point[1] * (height - 1)
    return x, y


def _rgb8(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in color)


def render_strokes(
    strokes: Iterable[Stroke],
    *,
    size: tuple[int, int],
    background: tuple[int, int, int] = (255, 255, 255),
    samples_per_curve: int = 32,
) -> Image.Image:
    """Render an ordered stroke program to an RGB Pillow image.

    Args:
        strokes: Ordered strokes; later strokes are composited over earlier ones.
        size: Output width and height in pixels.
        background: 8-bit RGB canvas background.
        samples_per_curve: Polyline samples used to approximate each Bezier curve.

    The renderer is deliberately simple and non-differentiable. Its only role in
    early Phase 0 is to validate the explicit stroke representation before we
    introduce diffvg or any learned model.
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if samples_per_curve < 2:
        raise ValueError("samples_per_curve must be at least 2")

    canvas = Image.new("RGB", size, background)

    for stroke in strokes:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        points = [
            _to_pixel(stroke.point_at(i / (samples_per_curve - 1)), width, height)
            for i in range(samples_per_curve)
        ]

        pixel_width = max(1, round(stroke.width * min(width, height)))
        alpha = round(stroke.opacity * 255)
        draw.line(points, fill=(*_rgb8(stroke.color), alpha), width=pixel_width, joint="curve")

        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    return canvas
