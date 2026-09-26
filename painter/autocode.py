"""Autonomous code-mutation harness for Neural Painter autoresearch.

This module turns a research diagnosis into an isolated coding-agent experiment:
create a git worktree, ask Codex to implement one bounded mutation, independently
run CI checks, rerun autoresearch, and keep only mutations that improve the
measured research frontier.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentRun:
    returncode: int
    stdout_path: Path
    stderr_path: Path


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def ensure_clean_repository(repo_root: Path) -> None:
    """Refuse autonomous mutation if the source repository has local edits."""
    result = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    if result.stdout.strip():
        raise RuntimeError(
            "repository has uncommitted changes; commit or stash them before autoresearch"
        )


def require_codex() -> str:
    """Resolve the Codex CLI used for local autonomous source editing."""
    executable = shutil.which(os.environ.get("NEURAL_PAINTER_CODEX", "codex"))
    if executable is None:
        raise RuntimeError(
            "Codex CLI was not found. Install/configure it or set "
            "NEURAL_PAINTER_CODEX to its executable path."
        )
    return executable


def build_agent_prompt(
    mutation: dict[str, str],
    *,
    baseline_summary: dict[str, Any],
) -> str:
    """Build a tightly scoped research-engineering prompt for the coding agent."""
    champion = baseline_summary.get("champion") or {}
    metrics = champion.get("metrics") or {}
    diagnoses = baseline_summary.get("final_diagnoses") or []

    return f"""You are modifying the Neural Painter research repository in an isolated git worktree.

Implement exactly ONE bounded research mutation. Do not redesign unrelated infrastructure.

HYPOTHESIS:
{mutation.get("hypothesis", "")}

REQUESTED CHANGE:
{mutation.get("change", "")}

PRIMARY AREA:
{mutation.get("area", "")}

CURRENT RECONSTRUCTION CHAMPION:
method={champion.get("candidate", {}).get("method")}
mse={metrics.get("mse")}
ssim={metrics.get("ssim")}
boundary_f1={metrics.get("diagnostics", {}).get("boundary_f1")}
mean_boundary_distance_px={metrics.get("diagnostics", {}).get("mean_boundary_distance_px")}
high_frequency_ratio={metrics.get("diagnostics", {}).get("high_frequency_ratio")}

ACTIVE DIAGNOSES:
{json.dumps(diagnoses, indent=2)}

Constraints:
- Keep existing public experiment methods working.
- Preserve deterministic seeds.
- Do not delete tests or weaken assertions merely to pass CI.
- Add or update focused tests for the mutation.
- Do not commit changes; the harness owns commits.
- You may inspect the repository and run focused tests.
- Prefer the smallest code change that directly tests the hypothesis.
- Do not edit generated outputs or research registries.

Finish with a concise explanation of what you changed and why.
"""


def invoke_codex(
    worktree: Path,
    prompt: str,
    *,
    log_dir: Path,
    timeout: int,
) -> AgentRun:
    """Run Codex non-interactively inside the isolated worktree."""
    executable = require_codex()
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "codex.jsonl"
    stderr_path = log_dir / "codex.stderr.txt"

    result = _run(
        [executable, "exec", "--json", "--full-auto", prompt],
        cwd=worktree,
        timeout=timeout,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return AgentRun(
        returncode=result.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def run_quality_gates(
    worktree: Path,
    *,
    timeout: int,
) -> tuple[bool, dict[str, str | int]]:
    """Independently validate an agent mutation before benchmarking it."""
    ruff = _run(["ruff", "check", "."], cwd=worktree, timeout=timeout)
    if ruff.returncode != 0:
        return False, {
            "gate": "ruff",
            "returncode": ruff.returncode,
            "stdout": ruff.stdout[-8000:],
            "stderr": ruff.stderr[-8000:],
        }

    tests = _run(["pytest", "-q"], cwd=worktree, timeout=timeout)
    if tests.returncode != 0:
        return False, {
            "gate": "pytest",
            "returncode": tests.returncode,
            "stdout": tests.stdout[-12000:],
            "stderr": tests.stderr[-12000:],
        }

    return True, {
        "gate": "passed",
        "returncode": 0,
        "stdout": tests.stdout[-4000:],
        "stderr": tests.stderr[-4000:],
    }


def _champion_metrics(summary: dict[str, Any]) -> dict[str, float]:
    category = summary.get("category_champions") or {}
    reconstruction = category.get("reconstruction") or summary.get("champion") or {}
    run = reconstruction.get("metrics") or {}
    diagnostics = run.get("diagnostics") or {}
    return {
        "mse": float(run.get("mse", float("inf"))),
        "ssim": float(run.get("ssim", 0.0)),
        "boundary_f1": float(diagnostics.get("boundary_f1", 0.0)),
        "boundary_distance": float(
            diagnostics.get("mean_boundary_distance_px", float("inf"))
        ),
        "high_frequency_ratio": float(
            diagnostics.get("high_frequency_ratio", float("inf"))
        ),
        "runtime_ms": float(run.get("render_ms", float("inf"))),
    }


def decide_acceptance(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    *,
    mse_regression_tolerance: float = 0.005,
    ssim_regression_tolerance: float = 0.002,
    runtime_multiplier_limit: float = 2.5,
) -> AcceptanceDecision:
    """Accept only measured frontier progress without material regressions."""
    before = _champion_metrics(baseline_summary)
    after = _champion_metrics(candidate_summary)
    reasons: list[str] = []

    if after["mse"] > before["mse"] * (1.0 + mse_regression_tolerance):
        reasons.append(
            f"MSE regressed from {before['mse']:.6f} to {after['mse']:.6f}"
        )
    if after["ssim"] < before["ssim"] - ssim_regression_tolerance:
        reasons.append(
            f"SSIM regressed from {before['ssim']:.6f} to {after['ssim']:.6f}"
        )
    if (
        before["runtime_ms"] < float("inf")
        and after["runtime_ms"] > before["runtime_ms"] * runtime_multiplier_limit
    ):
        reasons.append(
            "runtime exceeded allowed multiplier "
            f"({after['runtime_ms']:.0f} ms vs {before['runtime_ms']:.0f} ms)"
        )

    improved = (
        after["mse"] < before["mse"] * 0.999
        or after["ssim"] > before["ssim"] + 0.0005
        or after["boundary_f1"] > before["boundary_f1"] + 0.005
        or after["boundary_distance"] < before["boundary_distance"] - 0.1
        or abs(after["high_frequency_ratio"] - 1.0)
        < abs(before["high_frequency_ratio"] - 1.0) - 0.02
    )
    if not improved:
        reasons.append("no material reconstruction/structure/clutter metric improved")

    return AcceptanceDecision(accepted=not reasons, reasons=tuple(reasons))


def run_autoresearch_command(
    worktree: Path,
    image: str,
    output_root: Path,
    *,
    iterations: int,
    budget: int,
    palette_size: int,
    seed: int,
    device: str,
    timeout: int,
) -> dict[str, Any]:
    """Run the normal controller and return its summary."""
    command = [
        "python",
        "scripts/run_autoresearch.py",
        image,
        "--output-root",
        str(output_root),
        "--iterations",
        str(iterations),
        "--budget",
        str(budget),
        "--palette-size",
        str(palette_size),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    result = _run(command, cwd=worktree, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "autoresearch benchmark failed:\n"
            + (result.stderr[-12000:] or result.stdout[-12000:])
        )
    summary_path = output_root / "research_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"autoresearch did not write {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _git(worktree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = _run(["git", *args], cwd=worktree)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def autonomous_research(
    repo_root: Path,
    image: str,
    output_root: Path,
    *,
    mutation_cycles: int,
    experiment_iterations: int,
    budget: int,
    palette_size: int,
    seed: int,
    device: str,
    agent_timeout: int = 1800,
    gate_timeout: int = 900,
    experiment_timeout: int = 3600,
    apply: bool = False,
) -> dict[str, Any]:
    """Let diagnostics drive isolated coding-agent mutations and measured selection."""
    if mutation_cycles < 1:
        raise ValueError("mutation_cycles must be positive")
    ensure_clean_repository(repo_root)
    require_codex()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    branch = f"autoresearch/{stamp}"
    worktree = output_root / "_worktree"
    output_root.mkdir(parents=True, exist_ok=True)

    create = _run(
        ["git", "worktree", "add", "-b", branch, str(worktree), "HEAD"],
        cwd=repo_root,
    )
    if create.returncode != 0:
        raise RuntimeError(create.stderr.strip() or "failed to create autoresearch worktree")

    history: list[dict[str, Any]] = []
    try:
        baseline_root = output_root / "baseline"
        baseline = run_autoresearch_command(
            worktree,
            image,
            baseline_root,
            iterations=experiment_iterations,
            budget=budget,
            palette_size=palette_size,
            seed=seed,
            device=device,
            timeout=experiment_timeout,
        )

        for cycle in range(mutation_cycles):
            mutations = baseline.get("recommended_code_mutations") or []
            if not mutations:
                history.append(
                    {"cycle": cycle, "status": "stopped", "reason": "no mutation proposed"}
                )
                break

            mutation = mutations[0]
            cycle_dir = output_root / f"mutation_{cycle:02d}"
            prompt = build_agent_prompt(mutation, baseline_summary=baseline)
            (cycle_dir / "prompt.txt").parent.mkdir(parents=True, exist_ok=True)
            (cycle_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            agent = invoke_codex(
                worktree,
                prompt,
                log_dir=cycle_dir,
                timeout=agent_timeout,
            )
            if agent.returncode != 0:
                _git(worktree, "reset", "--hard", "HEAD")
                _git(worktree, "clean", "-fd")
                history.append(
                    {
                        "cycle": cycle,
                        "status": "rejected",
                        "reason": "coding agent failed",
                        "mutation": mutation,
                    }
                )
                continue

            diff = _git(worktree, "diff", "--stat").stdout.strip()
            if not diff:
                history.append(
                    {
                        "cycle": cycle,
                        "status": "rejected",
                        "reason": "coding agent produced no source changes",
                        "mutation": mutation,
                    }
                )
                continue

            gates_ok, gate_report = run_quality_gates(
                worktree,
                timeout=gate_timeout,
            )
            (cycle_dir / "quality_gates.json").write_text(
                json.dumps(gate_report, indent=2),
                encoding="utf-8",
            )
            if not gates_ok:
                _git(worktree, "reset", "--hard", "HEAD")
                _git(worktree, "clean", "-fd")
                history.append(
                    {
                        "cycle": cycle,
                        "status": "rejected",
                        "reason": f"quality gate failed: {gate_report['gate']}",
                        "mutation": mutation,
                    }
                )
                continue

            candidate_root = cycle_dir / "benchmark"
            candidate = run_autoresearch_command(
                worktree,
                image,
                candidate_root,
                iterations=experiment_iterations,
                budget=budget,
                palette_size=palette_size,
                seed=seed,
                device=device,
                timeout=experiment_timeout,
            )
            decision = decide_acceptance(baseline, candidate)
            comparison = {
                "baseline": _champion_metrics(baseline),
                "candidate": _champion_metrics(candidate),
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
            }
            (cycle_dir / "comparison.json").write_text(
                json.dumps(comparison, indent=2),
                encoding="utf-8",
            )

            if decision.accepted:
                _git(worktree, "add", "-A")
                _git(
                    worktree,
                    "commit",
                    "-m",
                    f"autoresearch: mutation {cycle + 1}",
                )
                baseline = candidate
                history.append(
                    {
                        "cycle": cycle,
                        "status": "accepted",
                        "mutation": mutation,
                        "comparison": comparison,
                    }
                )
            else:
                _git(worktree, "reset", "--hard", "HEAD")
                _git(worktree, "clean", "-fd")
                history.append(
                    {
                        "cycle": cycle,
                        "status": "rejected",
                        "mutation": mutation,
                        "comparison": comparison,
                    }
                )

        head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        result = {
            "branch": branch,
            "head": head,
            "history": history,
            "final_summary": baseline,
            "applied": False,
        }

        if apply:
            original_branch = _git(repo_root, "branch", "--show-current").stdout.strip()
            if not original_branch:
                raise RuntimeError("source repository is on a detached HEAD; cannot --apply")
            merge = _run(["git", "merge", "--ff-only", branch], cwd=repo_root)
            if merge.returncode != 0:
                raise RuntimeError(
                    "accepted autoresearch branch could not be fast-forwarded: "
                    + merge.stderr.strip()
                )
            result["applied"] = True
            result["applied_branch"] = original_branch

        result_path = output_root / "autocode_summary.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root)
