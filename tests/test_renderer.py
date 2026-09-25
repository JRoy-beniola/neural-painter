"""Tests for the deterministic Phase 0 stroke renderer."""

import pytest
from PIL import Image

from painter.renderer import render_strokes
from painter.stroke import Stroke


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
