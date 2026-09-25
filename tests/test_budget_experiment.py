"""End-to-end smoke test for the Phase 0 budget experiment."""

import json

import numpy as np
from PIL import Image

from painter.experiment import run_budget_experiment


def test_budget_experiment_writes_expected_artifacts(tmp_path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = (30, 60, 200)
    image[:, 8:] = (220, 180, 40)

    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "run"
    Image.fromarray(image, mode="RGB").save(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[3, 5],
    )

    assert report["budgets"] == [3, 5]
    assert len(report["runs"]) == 2
    assert (output_dir / "target.png").exists()
    assert (output_dir / "painted_3.png").exists()
    assert (output_dir / "painted_5.png").exists()
    assert (output_dir / "metrics.json").exists()

    stored = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert stored["seed"] == 7
    assert [run["stroke_count"] for run in stored["runs"]] == [3, 5]
