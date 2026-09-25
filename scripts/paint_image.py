"""Run the Phase 0 single-image stroke baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from painter.palette import extract_palette
from painter.renderer import render_strokes
from painter.sampling import sample_gradient_strokes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/painted.png"))
    parser.add_argument("--strokes", type=int, default=500)
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Image.open(args.image).convert("RGB")
    image_rgb = np.asarray(source, dtype=np.float32) / 255.0

    palette = extract_palette(image_rgb, args.palette_size, random_state=args.seed)
    strokes = sample_gradient_strokes(
        image_rgb,
        palette,
        args.strokes,
        seed=args.seed,
    )
    painted = render_strokes(strokes, size=source.size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    painted.save(args.output)
    print(f"wrote {args.output} with {len(strokes)} strokes and {len(palette)} palette colors")


if __name__ == "__main__":
    main()
