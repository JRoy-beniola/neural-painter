"""Stroke-space data model for the Phase 0 painter.

The representation is intentionally renderer-agnostic. Coordinates are normalized
to [0, 1] so the same stroke program can be rendered at arbitrary resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
RGB: TypeAlias = tuple[float, float, float]


def _validate_unit_interval(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _validate_point(point: Point, name: str) -> None:
    if len(point) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    _validate_unit_interval(point[0], f"{name}.x")
    _validate_unit_interval(point[1], f"{name}.y")


def _validate_rgb(color: RGB) -> None:
    if len(color) != 3:
        raise ValueError("color must contain exactly three channels")
    for index, channel in enumerate(color):
        _validate_unit_interval(channel, f"color[{index}]")


@dataclass(frozen=True, slots=True)
class Stroke:
    """A bounded quadratic Bezier brush stroke.

    Attributes:
        p0: Normalized start point.
        p1: Normalized quadratic Bezier control point.
        p2: Normalized end point.
        width: Normalized brush width in (0, 1].
        color: RGB color with channels in [0, 1].
        opacity: Stroke opacity in [0, 1].

    Ordering is represented by list position rather than a field on the stroke.
    That keeps the primitive minimal while still allowing later rendering stages
    to treat a stroke sequence as an ordered program.
    """

    p0: Point
    p1: Point
    p2: Point
    width: float
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        _validate_point(self.p0, "p0")
        _validate_point(self.p1, "p1")
        _validate_point(self.p2, "p2")

        if not isfinite(self.width):
            raise ValueError("width must be finite")
        if not 0.0 < self.width <= 1.0:
            raise ValueError("width must lie in (0, 1]")

        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")

    def point_at(self, t: float) -> Point:
        """Evaluate the quadratic Bezier centerline at normalized parameter t."""
        _validate_unit_interval(t, "t")

        one_minus_t = 1.0 - t
        x = (
            one_minus_t * one_minus_t * self.p0[0]
            + 2.0 * one_minus_t * t * self.p1[0]
            + t * t * self.p2[0]
        )
        y = (
            one_minus_t * one_minus_t * self.p0[1]
            + 2.0 * one_minus_t * t * self.p1[1]
            + t * t * self.p2[1]
        )
        return x, y



@dataclass(frozen=True, slots=True)
class TaperedStroke:
    """Quadratic Bezier stroke with linearly varying width."""

    p0: Point
    p1: Point
    p2: Point
    width_start: float
    width_mid: float
    width_end: float
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        _validate_point(self.p0, "p0")
        _validate_point(self.p1, "p1")
        _validate_point(self.p2, "p2")
        for name, value in (
            ("width_start", self.width_start),
            ("width_mid", self.width_mid),
            ("width_end", self.width_end),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")

    def point_at(self, t: float) -> Point:
        _validate_unit_interval(t, "t")
        one_minus_t = 1.0 - t
        return (
            one_minus_t * one_minus_t * self.p0[0]
            + 2.0 * one_minus_t * t * self.p1[0]
            + t * t * self.p2[0],
            one_minus_t * one_minus_t * self.p0[1]
            + 2.0 * one_minus_t * t * self.p1[1]
            + t * t * self.p2[1],
        )

    def width_at(self, t: float) -> float:
        _validate_unit_interval(t, "t")
        if t <= 0.5:
            local = t * 2.0
            return (1.0 - local) * self.width_start + local * self.width_mid
        local = (t - 0.5) * 2.0
        return (1.0 - local) * self.width_mid + local * self.width_end


@dataclass(frozen=True, slots=True)
class EllipsePatch:
    """Filled oriented ellipse for broad smooth image regions."""

    center: Point
    radius_x: float
    radius_y: float
    angle: float
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        _validate_point(self.center, "center")
        for name, value in (("radius_x", self.radius_x), ("radius_y", self.radius_y)):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not isfinite(self.angle):
            raise ValueError("angle must be finite")
        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")


@dataclass(frozen=True, slots=True)
class PolygonPatch:
    """Filled polygonal region primitive for arbitrary local image masses."""

    vertices: tuple[Point, ...]
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("vertices must contain at least three points")
        if len(self.vertices) > 32:
            raise ValueError("vertices must contain at most 32 points")
        for index, point in enumerate(self.vertices):
            _validate_point(point, f"vertices[{index}]")
        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")


@dataclass(frozen=True, slots=True)
class BezierRibbon:
    """Filled cubic Bezier ribbon with smoothly varying half-width."""

    p0: Point
    p1: Point
    p2: Point
    p3: Point
    width_start: float
    width_mid: float
    width_end: float
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        for name, point in (
            ("p0", self.p0),
            ("p1", self.p1),
            ("p2", self.p2),
            ("p3", self.p3),
        ):
            _validate_point(point, name)
        for name, value in (
            ("width_start", self.width_start),
            ("width_mid", self.width_mid),
            ("width_end", self.width_end),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")

    def point_at(self, t: float) -> Point:
        _validate_unit_interval(t, "t")
        u = 1.0 - t
        return (
            u**3 * self.p0[0]
            + 3.0 * u * u * t * self.p1[0]
            + 3.0 * u * t * t * self.p2[0]
            + t**3 * self.p3[0],
            u**3 * self.p0[1]
            + 3.0 * u * u * t * self.p1[1]
            + 3.0 * u * t * t * self.p2[1]
            + t**3 * self.p3[1],
        )

    def tangent_at(self, t: float) -> Point:
        _validate_unit_interval(t, "t")
        u = 1.0 - t
        return (
            3.0 * u * u * (self.p1[0] - self.p0[0])
            + 6.0 * u * t * (self.p2[0] - self.p1[0])
            + 3.0 * t * t * (self.p3[0] - self.p2[0]),
            3.0 * u * u * (self.p1[1] - self.p0[1])
            + 6.0 * u * t * (self.p2[1] - self.p1[1])
            + 3.0 * t * t * (self.p3[1] - self.p2[1]),
        )

    def width_at(self, t: float) -> float:
        _validate_unit_interval(t, "t")
        if t <= 0.5:
            local = 2.0 * t
            return (1.0 - local) * self.width_start + local * self.width_mid
        local = 2.0 * (t - 0.5)
        return (1.0 - local) * self.width_mid + local * self.width_end


@dataclass(frozen=True, slots=True)
class ClosedBezierRegion:
    """Filled smooth closed region built from cubic Bezier boundary segments."""

    segments: tuple[tuple[Point, Point, Point, Point], ...]
    color: RGB
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if len(self.segments) < 2:
            raise ValueError("segments must contain at least two cubic segments")
        if len(self.segments) > 24:
            raise ValueError("segments must contain at most 24 cubic segments")
        for segment_index, segment in enumerate(self.segments):
            if len(segment) != 4:
                raise ValueError("each segment must contain exactly four points")
            for point_index, point in enumerate(segment):
                _validate_point(
                    point,
                    f"segments[{segment_index}][{point_index}]",
                )
        for index in range(len(self.segments)):
            end = self.segments[index][3]
            start = self.segments[(index + 1) % len(self.segments)][0]
            if abs(end[0] - start[0]) > 1e-6 or abs(end[1] - start[1]) > 1e-6:
                raise ValueError("Bezier region segments must form a closed chain")
        _validate_rgb(self.color)
        _validate_unit_interval(self.opacity, "opacity")


Primitive: TypeAlias = Stroke | TaperedStroke | EllipsePatch | PolygonPatch | BezierRibbon | ClosedBezierRegion
