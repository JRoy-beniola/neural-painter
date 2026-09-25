"""Tests for differentiable local stroke refinement."""

import numpy as np

from painter.palette import extract_palette
from painter.refine import (
    refine_residual_strokes,
    refine_residual_strokes_global,
    refine_residual_strokes_staged,
)


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


def test_global_refinement_preserves_budget_and_bounds_geometry() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[5:19, 5:19] = (0.75, 0.9, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    from painter.iterative import paint_residual

    baseline, _ = paint_residual(image, palette, 8, seed=6)
    refined, background, stats = refine_residual_strokes_global(
        image,
        palette,
        8,
        seed=6,
        steps=3,
        optimized_stroke_count=4,
        optimization_resolution=24,
        objective="mse",
        optimize_geometry=True,
        geometry_bound=0.02,
        continuous_color=True,
    )

    assert len(refined) == 8
    assert background == (0, 0, 0)
    assert stats.refined_strokes == 4
    assert stats.optimize_geometry is True
    assert stats.geometry_bound == 0.02
    assert stats.continuous_color is True
    assert stats.final_loss <= stats.initial_loss

    for before, after in zip(baseline, refined, strict=True):
        for before_point, after_point in (
            (before.p0, after.p0),
            (before.p1, after.p1),
            (before.p2, after.p2),
        ):
            assert max(
                abs(before_value - after_value)
                for before_value, after_value in zip(before_point, after_point, strict=True)
            ) <= 0.020001


def test_global_refinement_can_optimize_all_strokes() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[4:16, 4:16] = (0.3, 0.65, 0.9)
    palette = extract_palette(image, 2, random_state=0)

    strokes, _, stats = refine_residual_strokes_global(
        image,
        palette,
        6,
        seed=7,
        steps=2,
        optimized_stroke_count=None,
        optimization_resolution=20,
        optimize_geometry=False,
        continuous_color=False,
    )

    assert len(strokes) == 6
    assert stats.refined_strokes == 6
    assert stats.optimize_geometry is False
    assert stats.continuous_color is False


def test_staged_refinement_progresses_across_disjoint_batches() -> None:
    image = np.zeros((24, 24, 3), dtype=np.float32)
    image[5:19, 5:19] = (0.7, 0.85, 1.0)
    palette = extract_palette(image, 2, random_state=0)

    strokes, background, stats = refine_residual_strokes_staged(
        image,
        palette,
        8,
        seed=8,
        stages=2,
        batch_size=3,
        steps_per_stage=2,
        optimization_resolution=24,
        optimize_geometry=True,
        geometry_bound=0.02,
        continuous_color=True,
    )

    assert len(strokes) == 8
    assert background == (0, 0, 0)
    assert stats.stages == 2
    assert stats.refined_strokes == 6
    assert stats.steps == 4
    assert stats.final_loss >= 0.0


def test_staged_refinement_stops_after_all_strokes_are_seen() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)
    image[4:16, 4:16] = (0.3, 0.6, 0.9)
    palette = extract_palette(image, 2, random_state=0)

    strokes, _, stats = refine_residual_strokes_staged(
        image,
        palette,
        5,
        seed=9,
        stages=5,
        batch_size=2,
        steps_per_stage=1,
        optimization_resolution=20,
        optimize_geometry=False,
        continuous_color=False,
    )

    assert len(strokes) == 5
    assert stats.refined_strokes == 5
    assert stats.stages == 3
