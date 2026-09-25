"""Run the Phase 0 stroke-budget experiment on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from painter.metrics import reconstruction_metrics
from painter.palette import extract_palette
from painter.renderer import render_strokes
from painter.sampling import sample_gradient_strokes

DEFAULT_BUDGETS = (100, 250, 500, 1000, 2000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0_budget"))
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    return parser.parse_args()


def run_experiment(
    image_path: Path,
    output_dir: Path,
    *,
    palette_size: int,
    seed: int,
    budgets: list[int] | tuple[int, ...],
) -> dict[str, object]:
    """Run fixed-budget reconstructions and persist images plus metrics."""
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("budgets must contain positive integers")

    source = Image.open(image_path).convert("RGB")
    target_rgb = np.asarray(source, dtype=np.float32) / 255.0
    palette = extract_palette(target_rgb, palette_size, random_state=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(output_dir / "target.png")

    runs: list[dict[str, float | int | str]] = []
    for budget in budgets:
        started = perf_counter()
        strokes = sample_gradient_strokes(target_rgb, palette, budget, seed=seed)
        painted = render_strokes(strokes, size=source.size)
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
                **metrics,
            }
        )

    report: dict[str, object] = {
        "input": str(image_path),
        "width": source.width,
        "height": source.height,
        "palette_size": palette_size,
        "seed": seed,
        "budgets": list(budgets),
        "runs": runs,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:
    args = parse_args()
    report = run_experiment(
        args.image,
        args.output_dir,
        palette_size=args.palette_size,
        seed=args.seed,
        budgets=args.budgets,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
