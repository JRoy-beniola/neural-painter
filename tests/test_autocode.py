"""Tests for autonomous code-mutation decision logic."""

from __future__ import annotations

from painter.autocode import build_agent_prompt, decide_acceptance


def _summary(
    *,
    mse: float,
    ssim: float,
    boundary_f1: float,
    boundary_distance: float,
    high_frequency_ratio: float,
    runtime_ms: float,
) -> dict:
    run = {
        "mse": mse,
        "ssim": ssim,
        "render_ms": runtime_ms,
        "diagnostics": {
            "boundary_f1": boundary_f1,
            "mean_boundary_distance_px": boundary_distance,
            "high_frequency_ratio": high_frequency_ratio,
        },
    }
    return {
        "champion": {
            "candidate": {"method": "adaptive_closed_region_refined"},
            "metrics": run,
        },
        "category_champions": {
            "reconstruction": {
                "candidate": {"method": "adaptive_closed_region_refined"},
                "metrics": run,
            }
        },
        "final_diagnoses": [{"code": "boundary_placement"}],
    }


def test_acceptance_keeps_material_reconstruction_improvement() -> None:
    before = _summary(
        mse=0.0105,
        ssim=0.848,
        boundary_f1=0.46,
        boundary_distance=5.0,
        high_frequency_ratio=2.1,
        runtime_ms=60_000.0,
    )
    after = _summary(
        mse=0.0102,
        ssim=0.849,
        boundary_f1=0.47,
        boundary_distance=4.8,
        high_frequency_ratio=2.0,
        runtime_ms=62_000.0,
    )

    decision = decide_acceptance(before, after)

    assert decision.accepted
    assert decision.reasons == ()


def test_acceptance_rejects_regression_without_compensation() -> None:
    before = _summary(
        mse=0.0105,
        ssim=0.848,
        boundary_f1=0.46,
        boundary_distance=5.0,
        high_frequency_ratio=2.1,
        runtime_ms=60_000.0,
    )
    after = _summary(
        mse=0.0110,
        ssim=0.844,
        boundary_f1=0.45,
        boundary_distance=5.2,
        high_frequency_ratio=2.2,
        runtime_ms=65_000.0,
    )

    decision = decide_acceptance(before, after)

    assert not decision.accepted
    assert decision.reasons


def test_agent_prompt_scopes_one_mutation() -> None:
    summary = _summary(
        mse=0.0105,
        ssim=0.848,
        boundary_f1=0.46,
        boundary_distance=5.0,
        high_frequency_ratio=2.1,
        runtime_ms=60_000.0,
    )
    mutation = {
        "area": "painter/rich.py",
        "hypothesis": "closed-region contour fitting is too crude",
        "change": "locally optimize closed-region control points",
    }

    prompt = build_agent_prompt(mutation, baseline_summary=summary)

    assert "Implement exactly ONE bounded research mutation" in prompt
    assert mutation["hypothesis"] in prompt
    assert mutation["change"] in prompt
    assert mutation["area"] in prompt
    assert "Do not commit changes" in prompt
