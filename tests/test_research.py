"""Tests for diagnostics and bounded autoresearch control."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from painter.diagnostics import image_diagnostics
from painter.research import (
    Diagnosis,
    ExperimentCandidate,
    diagnose_run,
    mutation_plan,
    pareto_front,
    run_autoresearch,
    select_candidate,
)


def test_image_diagnostics_identical_image_is_structurally_good() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[8:24, 8:24] = 1.0

    metrics = image_diagnostics(image, image, background_rgb=(0, 0, 0))

    assert metrics["foreground_iou"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["mean_boundary_distance_px"] == 0.0
    assert abs(float(metrics["edge_energy_ratio"]) - 1.0) < 1e-6


def test_diagnose_run_detects_clutter_and_structure_failure() -> None:
    run = {
        "mse": 0.02,
        "diagnostics": {
            "foreground_iou": 0.60,
            "boundary_f1": 0.40,
            "mean_boundary_distance_px": 6.0,
            "edge_energy_ratio": 2.0,
            "high_frequency_ratio": 2.5,
            "largest_residual_component_fraction": 0.12,
        },
    }

    codes = {item.code for item in diagnose_run(run)}

    assert "global_structure" in codes
    assert "boundary_placement" in codes
    assert "clutter" in codes
    assert "capacity_allocation" in codes


def test_select_candidate_avoids_refinement_after_surrogate_mismatch() -> None:
    diagnoses = [
        Diagnosis(
            code="surrogate_mismatch",
            severity=1.0,
            evidence="test",
            recommendation="test",
        )
    ]
    candidate = select_candidate(diagnoses, attempted_keys=set())

    assert candidate is not None
    assert "refined" not in candidate.method


def test_pareto_front_keeps_tradeoff_records() -> None:
    records = [
        {
            "report": {
                "runs": [
                    {
                        "mse": 0.01,
                        "ssim": 0.85,
                        "render_ms": 20.0,
                        "diagnostics": {
                            "mean_boundary_distance_px": 3.0,
                            "edge_energy_ratio": 1.1,
                        },
                    }
                ]
            }
        },
        {
            "report": {
                "runs": [
                    {
                        "mse": 0.02,
                        "ssim": 0.80,
                        "render_ms": 10.0,
                        "diagnostics": {
                            "mean_boundary_distance_px": 5.0,
                            "edge_energy_ratio": 1.4,
                        },
                    }
                ]
            }
        },
    ]

    frontier = pareto_front(records)

    assert len(frontier) == 2


def test_mutation_plan_maps_failure_to_code_area() -> None:
    plan = mutation_plan(
        [
            Diagnosis(
                code="capacity_allocation",
                severity=0.8,
                evidence="test",
                recommendation="test",
            )
        ]
    )

    assert plan
    assert plan[0]["area"] == "painter/rich.py"


def test_run_autoresearch_writes_registry_and_summary(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[6:18, 6:18] = (180, 220, 250)
    Image.fromarray(image, mode="RGB").save(image_path)

    output_root = tmp_path / "research"
    summary = run_autoresearch(
        image_path,
        output_root,
        iterations=2,
        budget=8,
        palette_size=2,
        seed=3,
        device="cpu",
    )

    assert summary["iterations_completed"] == 2
    assert (output_root / "registry.jsonl").exists()
    assert (output_root / "research_summary.json").exists()

    lines = (output_root / "registry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "diagnoses" in first
    assert "mutation_plan" in first

    attempted = {
        ExperimentCandidate(**json.loads(json.dumps(record["candidate"]))).key
        for record in map(json.loads, lines)
    }
    assert len(attempted) == 2
