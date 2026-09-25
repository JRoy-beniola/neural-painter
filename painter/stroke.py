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
