"""Tests for differentiable local stroke refinement."""

import numpy as np

from painter.palette import extract_palette
from painter.refine import refine_residual_strokes


def test_refinement_preserves_stroke_budget_and_improves_soft_loss() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[6:18, 6:18] = (0.75, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    strokes, background, stats = refine_residual_strokes(
        image,
        palette,
        8,
        seed=2,
        steps=3,
        max_refine_strokes=4,
        optimization_resolution=24,
    )

    assert len(strokes) == 8
    assert background == (0, 0, 0)
    assert stats.refined_strokes == 4
    assert stats.steps == 3
    assert stats.final_loss <= stats.initial_loss


def test_refinement_is_deterministic_for_fixed_seed() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[5:15, 5:15] = (0.25, 0.55, 0.85)
    palette = extract_palette(image, 2, random_state=0)

    left, _, left_stats = refine_residual_strokes(
        image,
        palette,
        6,
        seed=4,
        steps=2,
        max_refine_strokes=3,
        optimization_resolution=20,
    )
    right, _, right_stats = refine_residual_strokes(
        image,
        palette,
        6,
        seed=4,
        steps=2,
        max_refine_strokes=3,
        optimization_resolution=20,
    )

    assert left == right
    assert left_stats == right_stats


def test_refinement_keeps_geometry_fixed() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[6:18, 6:18] = (0.8, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    from painter.iterative import paint_residual

    baseline, _ = paint_residual(image, palette, 8, seed=3)
    refined, _, _ = refine_residual_strokes(
        image,
        palette,
        8,
        seed=3,
        steps=2,
        max_refine_strokes=4,
        optimization_resolution=24,
    )

    for before, after in zip(baseline, refined, strict=True):
        assert before.p0 == after.p0
        assert before.p1 == after.p1
        assert before.p2 == after.p2
        assert 0.004 <= after.width <= 0.035
        assert 0.35 <= after.opacity <= 0.95


def test_structure_aware_refinement_improves_composite_loss() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[5:19, 5:19] = (0.75, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    strokes, _, stats = refine_residual_strokes(
        image,
        palette,
        8,
        seed=5,
        steps=3,
        max_refine_strokes=4,
        optimization_resolution=24,
        objective="structure",
    )

    assert len(strokes) == 8
    assert stats.objective == "structure"
    assert stats.final_loss <= stats.initial_loss
