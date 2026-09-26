"""One-shot bounded code mutation loop for Neural Painter."""

from __future__ import annotations

import ast
import difflib
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
    function_start_line: int
    function_end_line: int
    source: str
    function_source: str


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
    function_source = "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
    return FunctionContext(
        path=path,
        function_name=function_name,
        start_line=start,
        end_line=end,
        function_start_line=node.lineno,
        function_end_line=node.end_lineno,
        source=excerpt,
        function_source=function_source,
    )


def build_mutation_prompt(
    target: FunctionContext,
    *,
    objective: str,
    baseline_metrics: dict[str, Any] | None = None,
    behavior_preserving: bool = False,
) -> str:
    """Build a small prompt for one bounded replacement-function proposal."""
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
- Return the complete replacement definition of {target.function_name}, starting with def.
- Return Python source only: no unified diff, no Markdown fences, no explanation.
- Do not include any other function or top-level code.

SOURCE EXCERPT ({target.path}:{target.start_line}-{target.end_line}):
{target.source}"""


def parse_replacement_function(
    text: str,
    *,
    function_name: str,
    original_source: str,
) -> str:
    """Parse one replacement function and require an unchanged signature."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    marker = f"def {function_name}("
    start = stripped.find(marker)
    if start < 0:
        raise ValueError(f"model did not return {function_name!r} source")
    if stripped[:start].strip():
        raise ValueError("model returned prose before the replacement function")

    replacement = stripped[start:].strip() + "\n"
    try:
        replacement_tree = ast.parse(replacement)
        original_tree = ast.parse(original_source)
    except SyntaxError as exc:
        raise ValueError(f"model returned invalid Python: {exc.msg}") from exc

    replacement_nodes = [
        node
        for node in replacement_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    original_nodes = [
        node
        for node in original_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(replacement_tree.body) != 1 or len(replacement_nodes) != 1:
        raise ValueError("model must return exactly the requested function")
    if replacement_nodes[0].name != function_name:
        raise ValueError("model returned the wrong function")
    if len(original_nodes) != 1:
        raise ValueError("original function source is malformed")
    if ast.dump(replacement_nodes[0].args) != ast.dump(original_nodes[0].args):
        raise ValueError("model changed the function signature")
    return replacement


def build_function_patch(
    repo_root: Path,
    target: FunctionContext,
    replacement: str,
) -> str:
    """Construct a valid unified diff deterministically from replacement source."""
    source_path = repo_root / target.path
    before_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    after_lines = (
        before_lines[: target.function_start_line - 1]
        + replacement_lines
        + before_lines[target.function_end_line :]
    )
    patch = "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{target.path}",
            tofile=f"b/{target.path}",
        )
    )
    if not patch:
        raise ValueError("model proposal did not change the target function")
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
    """Ask the model once for one bounded function replacement."""
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
                    "Return only the complete replacement Python function. "
                    "Do not browse, plan aloud, or request more context."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    replacement = parse_replacement_function(
        response,
        function_name=function_name,
        original_source=target.function_source,
    )
    patch = build_function_patch(repo_root, target, replacement)
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
