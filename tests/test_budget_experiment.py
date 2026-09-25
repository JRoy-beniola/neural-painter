"""End-to-end smoke tests for the Phase 0 budget experiment."""

import json

import numpy as np
from PIL import Image

from painter.experiment import run_budget_experiment


def _write_input(path) -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[6:18, 6:18] = (180, 220, 250)
    Image.fromarray(image, mode="RGB").save(path)


def test_static_budget_experiment_writes_expected_artifacts(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "static"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[3, 5],
        method="static",
    )

    assert report["method"] == "static"
    assert report["budgets"] == [3, 5]
    assert len(report["runs"]) == 2
    assert (output_dir / "target.png").exists()
    assert (output_dir / "painted_3.png").exists()
    assert (output_dir / "painted_5.png").exists()
    assert (output_dir / "metrics.json").exists()

    stored = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert stored["seed"] == 7
    assert stored["method"] == "static"
    assert [run["stroke_count"] for run in stored["runs"]] == [3, 5]


def test_residual_budget_experiment_uses_estimated_background(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "residual"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[6],
        method="residual",
    )

    assert report["method"] == "residual"
    assert report["runs"][0]["background_rgb"] == [0, 0, 0]
    assert (output_dir / "painted_6.png").exists()


def test_refined_budget_experiment_records_refinement_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "refined"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[4],
        method="refined",
    )

    assert report["method"] == "refined"
    refinement = report["runs"][0]["refinement"]
    assert refinement["refined_strokes"] == 4
    assert refinement["final_loss"] <= refinement["initial_loss"]
    assert (output_dir / "painted_4.png").exists()


def test_global_refined_budget_experiment_records_capacity_audit_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "global_refined"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[4],
        method="global_refined",
        optimized_stroke_count=3,
        optimize_geometry=True,
        geometry_bound=0.02,
        continuous_color=True,
    )

    assert report["method"] == "global_refined"
    refinement = report["runs"][0]["refinement"]
    assert refinement["refined_strokes"] == 3
    assert refinement["optimize_geometry"] is True
    assert refinement["geometry_bound"] == 0.02
    assert refinement["continuous_color"] is True
    assert refinement["final_loss"] <= refinement["initial_loss"]


def test_staged_refined_budget_experiment_records_stage_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "staged_refined"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[6],
        method="staged_refined",
        optimized_stroke_count=2,
        optimize_geometry=True,
        geometry_bound=0.02,
        continuous_color=True,
        refinement_stages=2,
    )

    assert report["method"] == "staged_refined"
    refinement = report["runs"][0]["refinement"]
    assert refinement["stages"] == 2
    assert refinement["refined_strokes"] == 4
    assert refinement["optimize_geometry"] is True


def test_staged_refined_budget_experiment_records_sweep_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "staged_refined_sweeps"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[6],
        method="staged_refined_structure",
        optimized_stroke_count=2,
        optimize_geometry=True,
        geometry_bound=0.02,
        continuous_color=True,
        refinement_stages=3,
        refinement_sweeps=2,
    )

    refinement = report["runs"][0]["refinement"]
    assert refinement["sweeps"] == 2
    assert refinement["stages"] == 6
    assert refinement["refined_strokes"] == 6


def test_rich_residual_budget_experiment_writes_output(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "rich"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[8],
        method="rich_residual",
    )

    assert report["method"] == "rich_residual"
    assert report["runs"][0]["stroke_count"] == 8
    assert (output_dir / "painted_8.png").exists()


def test_region_rich_residual_budget_experiment_writes_output(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "region_rich"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[8],
        method="region_rich_residual",
    )

    assert report["method"] == "region_rich_residual"
    assert report["runs"][0]["stroke_count"] == 8
    assert (output_dir / "painted_8.png").exists()


def test_region_rich_refined_budget_experiment_records_metadata(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "region_rich_refined"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=7,
        budgets=[8],
        method="region_rich_refined",
        optimized_stroke_count=4,
        geometry_bound=0.02,
    )

    assert report["method"] == "region_rich_refined"
    refinement = report["runs"][0]["refinement"]
    assert refinement["refined_strokes"] >= 1
    assert refinement["final_loss"] <= refinement["initial_loss"]


def test_region_rich_refined_structure_ordered_smoke(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_dir = tmp_path / "region_rich_ordered"
    _write_input(input_path)

    report = run_budget_experiment(
        input_path,
        output_dir,
        palette_size=2,
        seed=11,
        budgets=[8],
        method="region_rich_refined_structure",
        optimized_stroke_count=3,
        geometry_bound=0.02,
    )

    refinement = report["runs"][0]["refinement"]
    assert refinement["refined_strokes"] >= 1
    assert refinement["final_loss"] <= refinement["initial_loss"]
    assert (output_dir / "painted_8.png").exists()
