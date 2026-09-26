"""Tests for the deterministic Phase 0 stroke renderer."""

import pytest
from PIL import Image

from painter.calibration import (
    dense_rich_renderer_consistency,
    ordered_context_renderer_consistency,
    rich_renderer_consistency,
)
from painter.renderer import render_strokes
from painter.stroke import EllipsePatch, PolygonPatch, Stroke, TaperedStroke


def horizontal_stroke(**overrides: object) -> Stroke:
    values = {
        "p0": (0.1, 0.5),
        "p1": (0.5, 0.5),
        "p2": (0.9, 0.5),
        "width": 0.1,
        "color": (1.0, 0.0, 0.0),
        "opacity": 1.0,
    }
    values.update(overrides)
    return Stroke(**values)


def test_empty_program_returns_background() -> None:
    image = render_strokes([], size=(20, 10), background=(7, 8, 9))

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (20, 10)
    assert image.getpixel((0, 0)) == (7, 8, 9)


def test_opaque_stroke_changes_canvas() -> None:
    image = render_strokes([horizontal_stroke()], size=(100, 100))

    assert image.getpixel((50, 50))[0] > 200
    assert image.getpixel((50, 50))[1] < 50
    assert image.getpixel((50, 50))[2] < 50


def test_half_opacity_blends_with_white_background() -> None:
    image = render_strokes(
        [horizontal_stroke(opacity=0.5, color=(0.0, 0.0, 0.0))],
        size=(100, 100),
    )

    center = image.getpixel((50, 50))
    assert center[0] == pytest.approx(127, abs=2)
    assert center[1] == pytest.approx(127, abs=2)
    assert center[2] == pytest.approx(127, abs=2)


def test_later_strokes_are_composited_on_top() -> None:
    red = horizontal_stroke(color=(1.0, 0.0, 0.0))
    blue = horizontal_stroke(color=(0.0, 0.0, 1.0))

    image = render_strokes([red, blue], size=(100, 100))

    center = image.getpixel((50, 50))
    assert center[2] > 200
    assert center[0] < 50


@pytest.mark.parametrize("size", [(0, 10), (10, 0), (-1, 10)])
def test_invalid_image_size_is_rejected(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        render_strokes([], size=size)


def test_invalid_curve_sampling_is_rejected() -> None:
    with pytest.raises(ValueError):
        render_strokes([], size=(10, 10), samples_per_curve=1)


def test_renderer_supports_tapered_strokes_and_patches() -> None:
    patch = EllipsePatch(
        center=(0.5, 0.5),
        radius_x=0.2,
        radius_y=0.1,
        angle=0.4,
        color=(0.0, 1.0, 0.0),
        opacity=1.0,
    )
    stroke = TaperedStroke(
        p0=(0.2, 0.5),
        p1=(0.5, 0.3),
        p2=(0.8, 0.5),
        width_start=0.02,
        width_mid=0.08,
        width_end=0.02,
        color=(1.0, 0.0, 0.0),
        opacity=1.0,
    )

    image = render_strokes([patch, stroke], size=(100, 100))

    center = image.getpixel((50, 40))
    assert center != (255, 255, 255)


def test_rich_soft_renderer_tracks_raster_renderer() -> None:
    metrics = rich_renderer_consistency(size=(96, 96))

    assert metrics["mse"] < 0.01
    assert metrics["ssim"] > 0.85


def test_dense_rich_soft_renderer_reports_accumulated_mismatch() -> None:
    metrics = dense_rich_renderer_consistency(size=(64, 64))

    assert metrics["mse"] < 0.02
    assert metrics["ssim"] > 0.55


def test_ordered_context_renderer_tracks_raster_renderer() -> None:
    metrics = ordered_context_renderer_consistency(size=(64, 64))

    assert metrics["patch_stage"]["mse"] < 0.01
    assert metrics["patch_stage"]["ssim"] > 0.80
    assert metrics["stroke_stage"]["mse"] < 0.01
    assert metrics["stroke_stage"]["ssim"] > 0.85


def test_renderer_supports_polygon_patch() -> None:
    patch = PolygonPatch(
        vertices=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
        color=(0.0, 0.5, 1.0),
        opacity=1.0,
    )
    image = render_strokes([patch], size=(64, 64), background=(0, 0, 0))
    assert image.getpixel((32, 32)) != (0, 0, 0)
