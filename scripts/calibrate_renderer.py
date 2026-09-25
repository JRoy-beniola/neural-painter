"""Report real-vs-soft renderer consistency for rich primitives."""

from __future__ import annotations

import json

from painter.calibration import (
    dense_rich_renderer_consistency,
    ordered_context_renderer_consistency,
    rich_renderer_consistency,
)


def main() -> None:
    report = {
        "simple_full_soft": rich_renderer_consistency(size=(96, 96)),
        "dense_full_soft": dense_rich_renderer_consistency(size=(72, 72)),
        "ordered_context": ordered_context_renderer_consistency(size=(72, 72)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
