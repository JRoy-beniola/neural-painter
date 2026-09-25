"""Tests for the residual-driven painter."""

import numpy as np

from painter.iterative import paint_residual
from painter.palette import extract_palette


def test_residual_painter_preserves_requested_stroke_budget() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[8:24, 8:24] = (0.8, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    strokes, background = paint_residual(image, palette, 17, seed=3)

    assert len(strokes) == 17
    assert background == (0, 0, 0)


def test_residual_painter_is_deterministic_for_fixed_seed() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[6:18, 6:18] = (0.2, 0.6, 0.9)
    palette = extract_palette(image, 2, random_state=0)

    left, _ = paint_residual(image, palette, 12, seed=11)
    right, _ = paint_residual(image, palette, 12, seed=11)

    assert left == right
