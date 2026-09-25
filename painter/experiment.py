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
from painter.renderer import render_strokes
from painter.sampling import sample_gradient_strokes
from painter.structured import paint_structured_residual

DEFAULT_BUDGETS = (100, 250, 500, 1000, 2000)
METHODS = ("static", "residual", "structured")


def run_budget_experiment(
    image_path: Path,
    output_dir: Path,
    *,
    palette_size: int,
    seed: int,
    budgets: list[int] | tuple[int, ...],
    method: str = "static",
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
        elif method == "residual":
            strokes, background = paint_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )
        else:
            strokes, background = paint_structured_residual(
                target_rgb,
                palette,
                budget,
                seed=seed,
            )

        painted = render_strokes(strokes, size=source.size, background=background)
        elapsed_ms = (perf_counter() - started) * 1000.0

        painted_path = output_dir / f"painted_{budget}.png"
        painted.save(painted_path)
        painted_rgb = np.asarray(painted, dtype=np.float32) / 255.0

        metrics = reconstruction_metrics(target_rgb, painted_rgb)
        runs.append(
            {
                "stroke_count": budget,
                "render_ms": elapsed_ms,
                "output": painted_path.name,
                "background_rgb": list(background),
                **metrics,
            }
        )

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
