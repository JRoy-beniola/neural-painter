"""Tests for the structured residual painter."""

import numpy as np

from painter.palette import extract_palette
from painter.structured import paint_structured_residual


def test_structured_painter_preserves_requested_stroke_budget() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[7:25, 7:25] = (0.8, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    strokes, background = paint_structured_residual(image, palette, 19, seed=5)

    assert len(strokes) == 19
    assert background == (0, 0, 0)


def test_structured_painter_is_deterministic_for_fixed_seed() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[5:19, 5:19] = (0.2, 0.6, 0.9)
    palette = extract_palette(image, 2, random_state=0)

    left, _ = paint_structured_residual(image, palette, 13, seed=9)
    right, _ = paint_structured_residual(image, palette, 13, seed=9)

    assert left == right
