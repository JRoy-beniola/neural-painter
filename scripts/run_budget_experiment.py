"""Run the Phase 0 stroke-budget experiment on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from painter.experiment import DEFAULT_BUDGETS, METHODS, run_budget_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0_budget"))
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--method", choices=METHODS, default="static")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--optimized-stroke-count",
        type=int,
        default=256,
        help="Strokes to optimize in global refinement; use 0 for all strokes.",
    )
    parser.add_argument("--optimize-geometry", action="store_true")
    parser.add_argument("--geometry-bound", type=float, default=0.03)
    parser.add_argument("--continuous-color", action="store_true")
    parser.add_argument("--refinement-stages", type=int, default=4)
    parser.add_argument("--refinement-sweeps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_budget_experiment(
        args.image,
        args.output_dir,
        palette_size=args.palette_size,
        seed=args.seed,
        budgets=args.budgets,
        method=args.method,
        device=args.device,
        optimized_stroke_count=None if args.optimized_stroke_count == 0 else args.optimized_stroke_count,
        optimize_geometry=args.optimize_geometry,
        geometry_bound=args.geometry_bound,
        continuous_color=args.continuous_color,
        refinement_stages=args.refinement_stages,
        refinement_sweeps=args.refinement_sweeps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
