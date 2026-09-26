"""Reusable Phase 0 experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from painter.iterative import paint_residual
from painter.metrics import reconstruction_metrics
from painter.palette import extract_palette
from painter.refine import (
    refine_residual_strokes,
    refine_residual_strokes_global,
    refine_residual_strokes_staged,
)
from painter.renderer import render_strokes
from painter.rich import (
    paint_polygon_rich_residual,
    paint_region_rich_residual,
    paint_rich_residual,
)
from painter.rich_refine import (
    refine_polygon_rich_primitives_ordered,
    refine_region_rich_primitives_ordered,
    refine_region_rich_primitives_scheduled,
)
from painter.sampling import sample_gradient_strokes
from painter.structured import paint_structured_residual

DEFAULT_BUDGETS = (100, 250, 500, 1000, 2000)
METHODS = ("static", "residual", "structured", "rich_residual", "region_rich_residual", "polygon_rich_residual", "polygon_rich_refined_contour", "region_rich_refined", "region_rich_refined_structure", "region_rich_refined_schedule", "refined", "refined_structure", "global_refined", "global_refined_structure", "staged_refined", "staged_refined_structure")


def run_budget_experiment(
    image_path: Path,
    output_dir: Path,
    *,
    palette_size: int,
    seed: int,
    budgets: list[int] | tuple[int, ...],
    method: str = "static",
    device: str = "auto",
    optimized_stroke_count: int | None = 256,
    optimize_geometry: bool = False,
    geometry_bound: float = 0.03,
    continuous_color: bool = False,
    refinement_stages: int = 4,
    refinement_sweeps: int = 1,
    cleanup_stroke_count: int = 64,
    cleanup_geometry_bound: float = 0.0,
    cleanup_optimize_geometry: bool = False,
    cleanup_geometry_drift_weight: float = 0.10,
) -> dict[str, object]:
    """Run fixed-budget reconstructions and persist images plus metrics."""
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("budgets must contain positive integers")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")

    source = Image.open(image_path).convert("RGB")
    target_rgb = np.asarray(source, dtype=np.float32) / 255.0
    palette = extract_palette(target_rgb, palette_size, random_state=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(output_dir / "target.png")

    runs: list[dict[str, float | int | str | list[int]]] = []
    for budget in budgets:
        started = perf_counter()

        if method == "static":
            strokes = sample_gradient_strokes(target_rgb, palette, budget, seed=seed)
            background = (255, 255, 255)
            refinement = None
        elif method == "residual":
            strokes, background = paint_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
            refinement = None
        elif method == "structured":
            strokes, background = paint_structured_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
            refinement = None
        elif method == "rich_residual":
            strokes, background = paint_rich_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
            refinement = None
        elif method == "region_rich_residual":
            strokes, background = paint_region_rich_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
            refinement = None
        elif method == "polygon_rich_residual":
            strokes, background = paint_polygon_rich_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
            refinement = None
        elif method == "polygon_rich_refined_contour":
            strokes, background, refinement = refine_polygon_rich_primitives_ordered(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                max_refine_strokes=256 if optimized_stroke_count is None else optimized_stroke_count,
                geometry_bound=geometry_bound,
            )
        elif method == "region_rich_refined":
            strokes, background, refinement = refine_region_rich_primitives_ordered(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="mse",
                max_refine_strokes=256 if optimized_stroke_count is None else optimized_stroke_count,
                geometry_bound=geometry_bound,
            )
        elif method == "region_rich_refined_structure":
            strokes, background, refinement = refine_region_rich_primitives_ordered(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="structure",
                max_refine_strokes=256 if optimized_stroke_count is None else optimized_stroke_count,
                geometry_bound=geometry_bound,
            )
        elif method == "region_rich_refined_schedule":
            structure_strokes = 256 if optimized_stroke_count is None else optimized_stroke_count
            strokes, background, refinement = refine_region_rich_primitives_scheduled(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                structure_strokes=structure_strokes,
                cleanup_strokes=cleanup_stroke_count,
                structure_geometry_bound=geometry_bound,
                cleanup_geometry_bound=cleanup_geometry_bound,
                cleanup_optimize_geometry=cleanup_optimize_geometry,
                cleanup_geometry_drift_weight=cleanup_geometry_drift_weight,
            )
        elif method == "refined":
            strokes, background, refinement = refine_residual_strokes(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="mse",
            )
        elif method == "refined_structure":
            strokes, background, refinement = refine_residual_strokes(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="structure",
            )
        elif method == "global_refined":
            strokes, background, refinement = refine_residual_strokes_global(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="mse",
                optimized_stroke_count=optimized_stroke_count,
                optimize_geometry=optimize_geometry,
                geometry_bound=geometry_bound,
                continuous_color=continuous_color,
            )
        elif method == "global_refined_structure":
            strokes, background, refinement = refine_residual_strokes_global(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="structure",
                optimized_stroke_count=optimized_stroke_count,
                optimize_geometry=optimize_geometry,
                geometry_bound=geometry_bound,
                continuous_color=continuous_color,
            )
        elif method == "staged_refined":
            strokes, background, refinement = refine_residual_strokes_staged(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="mse",
                stages=refinement_stages,
                sweeps=refinement_sweeps,
                batch_size=256 if optimized_stroke_count is None else optimized_stroke_count,
                optimize_geometry=optimize_geometry,
                geometry_bound=geometry_bound,
                continuous_color=continuous_color,
            )
        else:
            strokes, background, refinement = refine_residual_strokes_staged(
                target_rgb,
                palette,
                budget,
                seed=seed,
                device=device,
                objective="structure",
                stages=refinement_stages,
                sweeps=refinement_sweeps,
                batch_size=256 if optimized_stroke_count is None else optimized_stroke_count,
                optimize_geometry=optimize_geometry,
                geometry_bound=geometry_bound,
                continuous_color=continuous_color,
            )

        painted = render_strokes(strokes, size=source.size, background=background)
        elapsed_ms = (perf_counter() - started) * 1000.0

        painted_path = output_dir / f"painted_{budget}.png"
        painted.save(painted_path)
        painted_rgb = np.asarray(painted, dtype=np.float32) / 255.0

        metrics = reconstruction_metrics(target_rgb, painted_rgb)
        run: dict[str, float | int | str | list[int] | dict[str, float | int]] = {
            "stroke_count": budget,
            "render_ms": elapsed_ms,
            "output": painted_path.name,
            "background_rgb": list(background),
            **metrics,
        }
        if refinement is not None:
            run["refinement"] = {
                "refined_strokes": refinement.refined_strokes,
                "steps": refinement.steps,
                "initial_loss": refinement.initial_loss,
                "final_loss": refinement.final_loss,
                "device": refinement.device,
                "objective": refinement.objective,
                "optimize_geometry": refinement.optimize_geometry,
                "geometry_bound": refinement.geometry_bound,
                "continuous_color": refinement.continuous_color,
                "stages": refinement.stages,
                "sweeps": refinement.sweeps,
            }
        runs.append(run)

    report: dict[str, object] = {
        "input": str(image_path),
        "width": source.width,
        "height": source.height,
        "palette_size": palette_size,
        "seed": seed,
        "method": method,
        "budgets": list(budgets),
        "runs": runs,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
