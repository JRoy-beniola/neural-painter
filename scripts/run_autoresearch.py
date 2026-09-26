"""Run Neural Painter's bounded self-diagnosing research loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from painter.research import run_autoresearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_autoresearch(
        args.image,
        args.output_root,
        iterations=args.iterations,
        budget=args.budget,
        palette_size=args.palette_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
