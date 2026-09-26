"""One-shot bounded code mutation loop for Neural Painter."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from painter.agent.model import OpenAICompatibleModel
from painter.agent.tools import ProjectTools, _paths_from_patch


@dataclass(frozen=True, slots=True)
class FunctionContext:
    path: str
    function_name: str
    start_line: int
    end_line: int
    source: str


@dataclass(frozen=True, slots=True)
class MutationProposal:
    model: str
    target: FunctionContext
    prompt: str
    patch: str


def extract_function_context(
    repo_root: Path,
    path: str,
    function_name: str,
    *,
    context_lines: int = 8,
) -> FunctionContext:
    """Extract one top-level function plus a small amount of neighboring source."""
    source_path = repo_root / path
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one top-level function named {function_name!r} in {path}"
        )

    node = matches[0]
    if node.end_lineno is None:
        raise ValueError(f"could not determine end line for {function_name!r}")

    lines = source.splitlines()
    start = max(1, node.lineno - context_lines)
    end = min(len(lines), node.end_lineno + context_lines)
    excerpt = "\n".join(lines[start - 1 : end]) + "\n"
    return FunctionContext(
        path=path,
        function_name=function_name,
        start_line=start,
        end_line=end,
        source=excerpt,
    )


def build_mutation_prompt(
    target: FunctionContext,
    *,
    objective: str,
    baseline_metrics: dict[str, Any] | None = None,
    behavior_preserving: bool = False,
) -> str:
    """Build a deliberately small prompt for one bounded code proposal."""
    metrics = baseline_metrics or {}
    constraint = (
        "The edit must preserve executable behavior. Change only a comment or docstring."
        if behavior_preserving
        else "Make one minimal, mechanistically interpretable implementation change."
    )
    rendered_metrics = "\n".join(
        f"- {name}: {value}" for name, value in sorted(metrics.items())
    ) or "- none supplied"

    return f"""You are proposing one bounded edit to Neural Painter.

TARGET FILE:
{target.path}

TARGET FUNCTION:
{target.function_name}

OBJECTIVE:
{objective}

BASELINE METRICS:
{rendered_metrics}

CONSTRAINTS:
- Modify only {target.path}.
- Modify only the supplied function or its immediately adjacent comment/docstring.
- Preserve the function signature and all public APIs.
- Do not change evaluators, metrics, seeds, budgets, or experiment configuration.
- {constraint}
- Return exactly one standard unified diff beginning with --- a/{target.path}.
- Do not use Markdown fences.
- Do not explain the patch before or after the diff.

SOURCE EXCERPT ({target.path}:{target.start_line}-{target.end_line}):
{target.source}"""


def parse_unified_diff(text: str) -> str:
    """Extract a single unified diff while rejecting explanatory prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("--- a/")
    if start < 0:
        raise ValueError("model did not return a unified diff")
    if stripped[:start].strip():
        raise ValueError("model returned prose before the unified diff")

    patch = stripped[start:].strip() + "\n"
    if "\n+++ b/" not in patch or "\n@@" not in patch:
        raise ValueError("model diff is missing required unified-diff headers")
    return patch


def validate_target_patch(
    repo_root: Path,
    patch: str,
    *,
    target_path: str,
) -> None:
    """Require exactly one touched file and verify git can apply the patch."""
    touched = _paths_from_patch(patch)
    if touched != {target_path}:
        raise ValueError(
            f"mutation must touch only {target_path}; model touched {sorted(touched)}"
        )

    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=repo_root,
        input=patch,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if check.returncode != 0:
        raise ValueError((check.stderr or check.stdout).strip() or "patch does not apply")


def propose_function_patch(
    model: OpenAICompatibleModel,
    repo_root: Path,
    *,
    path: str,
    function_name: str,
    objective: str,
    baseline_metrics: dict[str, Any] | None = None,
    behavior_preserving: bool = False,
) -> MutationProposal:
    """Ask the model once for one bounded patch and validate it without applying it."""
    target = extract_function_context(repo_root, path, function_name)
    prompt = build_mutation_prompt(
        target,
        objective=objective,
        baseline_metrics=baseline_metrics,
        behavior_preserving=behavior_preserving,
    )
    response = model.complete_text(
        [
            {
                "role": "system",
                "content": (
                    "Return only the requested unified diff. "
                    "Do not browse, plan aloud, or request more context."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    patch = parse_unified_diff(response)
    validate_target_patch(repo_root, patch, target_path=path)
    return MutationProposal(
        model=model.model,
        target=target,
        prompt=prompt,
        patch=patch,
    )


def apply_proposal(repo_root: Path, proposal: MutationProposal) -> str:
    """Apply a previously validated proposal through the existing project sandbox."""
    return ProjectTools(repo_root).apply_patch(proposal.patch)
