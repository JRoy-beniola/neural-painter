"""Self-diagnosing bounded research controller for Neural Painter."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from painter.experiment import run_budget_experiment


@dataclass(frozen=True, slots=True)
class ExperimentCandidate:
    """One bounded painter experiment configuration."""

    method: str
    optimized_stroke_count: int | None = 256
    geometry_bound: float = 0.02
    cleanup_stroke_count: int = 64
    cleanup_geometry_bound: float = 0.0
    cleanup_optimize_geometry: bool = False

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Diagnosis:
    code: str
    severity: float
    evidence: str
    recommendation: str


def diagnose_run(run: dict[str, Any]) -> list[Diagnosis]:
    """Map measured failure signatures to explicit research diagnoses."""
    diagnostics = run.get("diagnostics", {})
    findings: list[Diagnosis] = []

    foreground_iou = float(diagnostics.get("foreground_iou", 1.0))
    boundary_f1 = float(diagnostics.get("boundary_f1", 1.0))
    boundary_distance = float(diagnostics.get("mean_boundary_distance_px", 0.0))
    edge_ratio = float(diagnostics.get("edge_energy_ratio", 1.0))
    frequency_ratio = float(diagnostics.get("high_frequency_ratio", 1.0))
    residual_fraction = float(
        diagnostics.get("largest_residual_component_fraction", 0.0)
    )

    refinement = run.get("refinement")
    if refinement:
        initial = float(refinement.get("initial_loss", 0.0))
        final = float(refinement.get("final_loss", initial))
        relative_drop = (initial - final) / max(abs(initial), 1e-8)
        if relative_drop > 0.25 and float(run.get("mse", 0.0)) > 0.012:
            findings.append(
                Diagnosis(
                    code="surrogate_mismatch",
                    severity=min(relative_drop, 1.0),
                    evidence=(
                        f"internal objective dropped {relative_drop:.1%} while "
                        f"final raster MSE remained {float(run.get('mse', 0.0)):.5f}"
                    ),
                    recommendation=(
                        "prefer unrefined or conservative variants; inspect loss/raster "
                        "alignment before adding optimization"
                    ),
                )
            )

    if foreground_iou < 0.82:
        findings.append(
            Diagnosis(
                code="global_structure",
                severity=min((0.82 - foreground_iou) / 0.35, 1.0),
                evidence=f"foreground IoU is only {foreground_iou:.3f}",
                recommendation=(
                    "increase broad-region capacity or change primitive allocation before cleanup"
                ),
            )
        )

    if boundary_f1 < 0.60 or boundary_distance > 4.0:
        findings.append(
            Diagnosis(
                code="boundary_placement",
                severity=min(max((0.60 - boundary_f1) / 0.35, boundary_distance / 12.0), 1.0),
                evidence=(
                    f"boundary F1={boundary_f1:.3f}, mean symmetric boundary "
                    f"distance={boundary_distance:.2f}px"
                ),
                recommendation=(
                    "favor region/ribbon initialization over more detail strokes; only use "
                    "contour refinement if it improves held-out raster metrics"
                ),
            )
        )

    if edge_ratio > 1.35 or frequency_ratio > 1.50:
        severity = max((edge_ratio - 1.0) / 1.5, (frequency_ratio - 1.0) / 2.0)
        findings.append(
            Diagnosis(
                code="clutter",
                severity=min(severity, 1.0),
                evidence=(
                    f"edge-energy ratio={edge_ratio:.2f}, high-frequency ratio="
                    f"{frequency_ratio:.2f}"
                ),
                recommendation=(
                    "reduce refinement/detail authority and prefer broad region primitives"
                ),
            )
        )

    if residual_fraction > 0.08:
        findings.append(
            Diagnosis(
                code="capacity_allocation",
                severity=min(residual_fraction / 0.25, 1.0),
                evidence=(
                    "largest high-error connected region occupies "
                    f"{residual_fraction:.1%} of the image"
                ),
                recommendation=(
                    "allocate more budget to polygons/ribbons rather than local cleanup"
                ),
            )
        )

    if not findings:
        findings.append(
            Diagnosis(
                code="fine_fidelity",
                severity=0.25,
                evidence="no dominant structural failure signature exceeded thresholds",
                recommendation=(
                    "run conservative local ablations and retain only Pareto improvements"
                ),
            )
        )

    return sorted(findings, key=lambda item: item.severity, reverse=True)


def candidate_library() -> tuple[ExperimentCandidate, ...]:
    """Safe search space over already-tested painter mechanisms."""
    return (
        ExperimentCandidate("adaptive_rich_residual", optimized_stroke_count=None),
        ExperimentCandidate("polygon_rich_residual", optimized_stroke_count=None),
        ExperimentCandidate("mixed_rich_residual", optimized_stroke_count=None),
        ExperimentCandidate("region_rich_residual", optimized_stroke_count=None),
        ExperimentCandidate(
            "polygon_rich_refined_contour",
            optimized_stroke_count=128,
            geometry_bound=0.01,
        ),
        ExperimentCandidate(
            "polygon_rich_refined_contour",
            optimized_stroke_count=256,
            geometry_bound=0.02,
        ),
        ExperimentCandidate(
            "mixed_rich_refined_positional",
            optimized_stroke_count=128,
            geometry_bound=0.01,
        ),
        ExperimentCandidate(
            "mixed_rich_refined_positional",
            optimized_stroke_count=256,
            geometry_bound=0.02,
        ),
        ExperimentCandidate(
            "region_rich_refined_structure",
            optimized_stroke_count=128,
            geometry_bound=0.01,
        ),
        ExperimentCandidate(
            "region_rich_refined_structure",
            optimized_stroke_count=256,
            geometry_bound=0.02,
        ),
        ExperimentCandidate(
            "region_rich_refined_schedule",
            optimized_stroke_count=256,
            geometry_bound=0.02,
            cleanup_stroke_count=64,
            cleanup_geometry_bound=0.0,
            cleanup_optimize_geometry=False,
        ),
    )


def _candidate_priority(
    candidate: ExperimentCandidate,
    diagnoses: list[Diagnosis],
) -> float:
    score = 0.0
    codes = {diagnosis.code for diagnosis in diagnoses}

    if "global_structure" in codes or "capacity_allocation" in codes:
        if candidate.method == "adaptive_rich_residual":
            score += 6.0
        if candidate.method == "polygon_rich_residual":
            score += 5.0
        if candidate.method == "mixed_rich_residual":
            score += 4.0
        if candidate.method == "region_rich_residual":
            score += 2.0

    if "clutter" in codes:
        if "residual" in candidate.method:
            score += 4.0
        if candidate.optimized_stroke_count is not None:
            score -= candidate.optimized_stroke_count / 128.0

    if "surrogate_mismatch" in codes:
        if "refined" in candidate.method:
            score -= 5.0
        else:
            score += 4.0

    if "boundary_placement" in codes and "polygon" in candidate.method:
        score += 2.0
    if "fine_fidelity" in codes and "refined" in candidate.method:
        score += 1.0

    score -= 0.1 * candidate.geometry_bound
    return score


def select_candidate(
    diagnoses: list[Diagnosis],
    *,
    attempted_keys: set[str],
) -> ExperimentCandidate | None:
    """Select the highest-priority unexplored bounded experiment."""
    unexplored = [
        candidate
        for candidate in candidate_library()
        if candidate.key not in attempted_keys
    ]
    if not unexplored:
        return None
    return max(unexplored, key=lambda item: _candidate_priority(item, diagnoses))


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated records over quality, structure, clutter, and runtime."""
    def vector(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
        run = record["report"]["runs"][0]
        diagnostics = run.get("diagnostics", {})
        return (
            float(run.get("mse", float("inf"))),
            -float(run.get("ssim", 0.0)),
            float(diagnostics.get("mean_boundary_distance_px", float("inf"))),
            abs(float(diagnostics.get("edge_energy_ratio", 1.0)) - 1.0),
            float(run.get("render_ms", float("inf"))),
        )

    frontier: list[dict[str, Any]] = []
    for index, candidate in enumerate(records):
        candidate_vector = vector(candidate)
        dominated = False
        for other_index, other in enumerate(records):
            if other_index == index:
                continue
            other_vector = vector(other)
            no_worse = all(a <= b for a, b in zip(other_vector, candidate_vector, strict=True))
            strictly_better = any(
                a < b for a, b in zip(other_vector, candidate_vector, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def champion_record(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose a stable champion from the current Pareto frontier.

    The champion is selected lexicographically: pixel fidelity first, then SSIM,
    boundary placement, edge-energy mismatch, and runtime.
    """
    frontier = pareto_front(records)
    if not frontier:
        return None

    def key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
        run = record["report"]["runs"][0]
        diagnostics = run.get("diagnostics", {})
        return (
            float(run.get("mse", float("inf"))),
            -float(run.get("ssim", 0.0)),
            float(diagnostics.get("mean_boundary_distance_px", float("inf"))),
            abs(float(diagnostics.get("edge_energy_ratio", 1.0)) - 1.0),
            float(run.get("render_ms", float("inf"))),
        )

    return min(frontier, key=key)


def frontier_diagnoses(
    records: list[dict[str, Any]],
) -> tuple[list[Diagnosis], list[Diagnosis]]:
    """Return champion diagnoses and failures persistent across the Pareto frontier."""
    frontier = pareto_front(records)
    champion = champion_record(records)
    champion_findings = (
        diagnose_run(champion["report"]["runs"][0])
        if champion is not None
        else []
    )
    if not frontier:
        return champion_findings, []

    findings_by_record = [
        diagnose_run(record["report"]["runs"][0])
        for record in frontier
    ]
    threshold = max(1, (len(frontier) + 1) // 2)
    counts: dict[str, int] = {}
    severities: dict[str, list[float]] = {}
    examples: dict[str, Diagnosis] = {}
    for findings in findings_by_record:
        seen: set[str] = set()
        for finding in findings:
            if finding.code in seen:
                continue
            seen.add(finding.code)
            counts[finding.code] = counts.get(finding.code, 0) + 1
            severities.setdefault(finding.code, []).append(finding.severity)
            current = examples.get(finding.code)
            if current is None or finding.severity > current.severity:
                examples[finding.code] = finding

    persistent: list[Diagnosis] = []
    for code, count in counts.items():
        if count < threshold:
            continue
        example = examples[code]
        persistent.append(
            Diagnosis(
                code=code,
                severity=float(np.mean(severities[code])),
                evidence=(
                    f"persistent on {count}/{len(frontier)} Pareto-front runs; "
                    f"example: {example.evidence}"
                ),
                recommendation=example.recommendation,
            )
        )
    persistent.sort(key=lambda item: item.severity, reverse=True)
    return champion_findings, persistent


def merge_diagnoses(
    champion_findings: list[Diagnosis],
    persistent_findings: list[Diagnosis],
) -> list[Diagnosis]:
    """Merge champion and persistent evidence without duplicating failure codes."""
    merged: dict[str, Diagnosis] = {}
    for finding in [*persistent_findings, *champion_findings]:
        current = merged.get(finding.code)
        if current is None or finding.severity > current.severity:
            merged[finding.code] = finding
    return sorted(merged.values(), key=lambda item: item.severity, reverse=True)


def mutation_plan(diagnoses: list[Diagnosis]) -> list[dict[str, str]]:
    """Translate unresolved diagnoses into bounded code-level research proposals."""
    proposals: list[dict[str, str]] = []
    codes = {diagnosis.code for diagnosis in diagnoses}

    if "global_structure" in codes or "capacity_allocation" in codes:
        proposals.append(
            {
                "area": "painter/rich.py",
                "hypothesis": "broad residual regions are underrepresented",
                "change": (
                    "use the adaptive allocator and extend it only if frontier evidence "
                    "shows coherent residual regions remain underrepresented"
                ),
            }
        )
    if "clutter" in codes:
        proposals.append(
            {
                "area": "painter/refine.py",
                "hypothesis": "detail optimization creates excess high-frequency contours",
                "change": (
                    "add an explicit contour-economy regularizer and ablate it independently"
                ),
            }
        )
    if "surrogate_mismatch" in codes:
        proposals.append(
            {
                "area": "painter/rich_refine.py",
                "hypothesis": "surrogate optimization is not transferring to the final raster",
                "change": (
                    "real-raster acceptance is now mandatory; inspect any remaining "
                    "surrogate mismatch only after rejected refinements are excluded"
                ),
            }
        )
    if "boundary_placement" in codes:
        proposals.append(
            {
                "area": "painter/rich.py",
                "hypothesis": "region initialization does not place smooth boundaries accurately",
                "change": (
                    "fit closed spline regions or locally optimize region vertices before detail"
                ),
            }
        )
    return proposals


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_autoresearch(
    image_path: Path,
    output_root: Path,
    *,
    iterations: int,
    budget: int,
    palette_size: int = 8,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    """Run diagnose -> hypothesize -> bounded experiment -> compare repeatedly."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    registry_path = output_root / "registry.jsonl"

    records: list[dict[str, Any]] = []
    attempted: set[str] = set()
    diagnoses = [
        Diagnosis(
            code="bootstrap",
            severity=1.0,
            evidence="no experiment has been run yet",
            recommendation="start from a strong broad-region baseline",
        )
    ]

    for iteration in range(iterations):
        candidate = select_candidate(diagnoses, attempted_keys=attempted)
        if candidate is None:
            break
        attempted.add(candidate.key)

        run_dir = output_root / f"{iteration:03d}_{candidate.method}"
        report = run_budget_experiment(
            image_path,
            run_dir,
            palette_size=palette_size,
            seed=seed,
            budgets=[budget],
            method=candidate.method,
            device=device,
            optimized_stroke_count=candidate.optimized_stroke_count,
            geometry_bound=candidate.geometry_bound,
            cleanup_stroke_count=candidate.cleanup_stroke_count,
            cleanup_geometry_bound=candidate.cleanup_geometry_bound,
            cleanup_optimize_geometry=candidate.cleanup_optimize_geometry,
        )
        run = report["runs"][0]
        last_run_diagnoses = diagnose_run(run)
        record = {
            "iteration": iteration,
            "timestamp": time(),
            "candidate": asdict(candidate),
            "report": report,
            "diagnoses": [asdict(item) for item in last_run_diagnoses],
            "mutation_plan": mutation_plan(last_run_diagnoses),
        }
        records.append(record)

        champion_findings, persistent_findings = frontier_diagnoses(records)
        diagnoses = merge_diagnoses(champion_findings, persistent_findings)
        record["frontier_diagnoses"] = [asdict(item) for item in diagnoses]
        _append_jsonl(registry_path, record)

    frontier = pareto_front(records)
    champion = champion_record(records)
    champion_findings, persistent_findings = frontier_diagnoses(records)
    active_findings = merge_diagnoses(champion_findings, persistent_findings)
    summary = {
        "image": str(image_path),
        "iterations_completed": len(records),
        "budget": budget,
        "pareto_front": [
            {
                "iteration": record["iteration"],
                "candidate": record["candidate"],
                "metrics": record["report"]["runs"][0],
            }
            for record in frontier
        ],
        "champion": (
            {
                "iteration": champion["iteration"],
                "candidate": champion["candidate"],
                "metrics": champion["report"]["runs"][0],
            }
            if champion is not None
            else None
        ),
        "champion_diagnoses": [asdict(item) for item in champion_findings],
        "persistent_frontier_diagnoses": [
            asdict(item) for item in persistent_findings
        ],
        "final_diagnoses": [asdict(item) for item in active_findings],
        "recommended_code_mutations": (
            mutation_plan(active_findings) if records else []
        ),
        "registry": str(registry_path),
    }
    with (output_root / "research_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
