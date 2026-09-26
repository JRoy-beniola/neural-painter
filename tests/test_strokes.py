"""Tests for the renderer-independent stroke-space contract."""

import math
from dataclasses import FrozenInstanceError

import pytest

from painter.stroke import (
    BezierRibbon,
    ClosedBezierRegion,
    EllipsePatch,
    PolygonPatch,
    Stroke,
    TaperedStroke,
)


def make_stroke(**overrides: object) -> Stroke:
    values = {
        "p0": (0.1, 0.2),
        "p1": (0.5, 0.8),
        "p2": (0.9, 0.3),
        "width": 0.05,
        "color": (0.2, 0.4, 0.6),
        "opacity": 0.75,
    }
    values.update(overrides)
    return Stroke(**values)


def test_valid_stroke_is_constructed() -> None:
    stroke = make_stroke()

    assert stroke.p0 == (0.1, 0.2)
    assert stroke.width == 0.05
    assert stroke.color == (0.2, 0.4, 0.6)
    assert stroke.opacity == 0.75


@pytest.mark.parametrize(
    "field,value",
    [
        ("p0", (-0.01, 0.5)),
        ("p1", (0.5, 1.01)),
        ("p2", (math.nan, 0.5)),
    ],
)
def test_control_points_must_be_finite_and_normalized(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_stroke(**{field: value})


@pytest.mark.parametrize("width", [0.0, -0.1, 1.01, math.inf, math.nan])
def test_width_must_be_positive_finite_and_normalized(width: float) -> None:
    with pytest.raises(ValueError):
        make_stroke(width=width)


@pytest.mark.parametrize(
    "color",
    [
        (-0.01, 0.5, 0.5),
        (0.5, 1.01, 0.5),
        (0.5, 0.5, math.nan),
    ],
)
def test_color_channels_must_be_finite_and_normalized(color: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        make_stroke(color=color)


@pytest.mark.parametrize("opacity", [-0.01, 1.01, math.inf, math.nan])
def test_opacity_must_be_finite_and_normalized(opacity: float) -> None:
    with pytest.raises(ValueError):
        make_stroke(opacity=opacity)


def test_point_at_hits_bezier_endpoints() -> None:
    stroke = make_stroke()

    assert stroke.point_at(0.0) == stroke.p0
    assert stroke.point_at(1.0) == stroke.p2


def test_point_at_matches_quadratic_bezier_midpoint() -> None:
    stroke = make_stroke(p0=(0.0, 0.0), p1=(0.5, 1.0), p2=(1.0, 0.0))

    assert stroke.point_at(0.5) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("t", [-0.01, 1.01, math.nan])
def test_point_at_requires_normalized_parameter(t: float) -> None:
    with pytest.raises(ValueError):
        make_stroke().point_at(t)


def test_stroke_is_immutable() -> None:
    stroke = make_stroke()

    with pytest.raises(FrozenInstanceError):
        stroke.width = 0.2


def test_tapered_stroke_interpolates_width() -> None:
    stroke = TaperedStroke(
        p0=(0.1, 0.5),
        p1=(0.5, 0.5),
        p2=(0.9, 0.5),
        width_start=0.01,
        width_mid=0.05,
        width_end=0.02,
        color=(0.2, 0.4, 0.6),
        opacity=0.8,
    )

    assert stroke.width_at(0.0) == pytest.approx(0.01)
    assert stroke.width_at(0.5) == pytest.approx(0.05)
    assert stroke.width_at(1.0) == pytest.approx(0.02)


def test_ellipse_patch_validates_bounds() -> None:
    patch = EllipsePatch(
        center=(0.5, 0.5),
        radius_x=0.1,
        radius_y=0.05,
        angle=0.3,
        color=(0.2, 0.4, 0.6),
        opacity=0.75,
    )
    assert patch.center == (0.5, 0.5)

    with pytest.raises(ValueError):
        EllipsePatch(
            center=(0.5, 0.5),
            radius_x=0.0,
            radius_y=0.05,
            angle=0.0,
            color=(0.2, 0.4, 0.6),
        )


def test_polygon_patch_validates_vertices() -> None:
    patch = PolygonPatch(
        vertices=((0.2, 0.2), (0.8, 0.2), (0.5, 0.8)),
        color=(0.2, 0.4, 0.7),
        opacity=0.8,
    )
    assert len(patch.vertices) == 3

    with pytest.raises(ValueError):
        PolygonPatch(
            vertices=((0.2, 0.2), (0.8, 0.2)),
            color=(0.2, 0.4, 0.7),
        )


def test_bezier_ribbon_validates_and_evaluates() -> None:
    ribbon = BezierRibbon(
        p0=(0.1, 0.5),
        p1=(0.3, 0.2),
        p2=(0.7, 0.8),
        p3=(0.9, 0.5),
        width_start=0.02,
        width_mid=0.08,
        width_end=0.03,
        color=(0.3, 0.6, 0.9),
        opacity=0.8,
    )
    assert ribbon.point_at(0.0) == ribbon.p0
    assert ribbon.point_at(1.0) == ribbon.p3
    assert ribbon.width_at(0.5) == ribbon.width_mid



def test_closed_bezier_region_validates_closed_chain() -> None:
    region = ClosedBezierRegion(
        segments=(
            ((0.2, 0.2), (0.4, 0.1), (0.6, 0.1), (0.8, 0.2)),
            ((0.8, 0.2), (0.9, 0.4), (0.9, 0.6), (0.8, 0.8)),
            ((0.8, 0.8), (0.6, 0.9), (0.4, 0.9), (0.2, 0.8)),
            ((0.2, 0.8), (0.1, 0.6), (0.1, 0.4), (0.2, 0.2)),
        ),
        color=(0.2, 0.5, 0.8),
        opacity=0.9,
    )
    assert len(region.segments) == 4
