"""Report real-vs-soft renderer consistency for rich primitives."""

from __future__ import annotations

import json

from painter.calibration import rich_renderer_consistency


def main() -> None:
    metrics = rich_renderer_consistency(size=(96, 96))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
