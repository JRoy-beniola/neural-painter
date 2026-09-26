"""Constrained repository tools exposed to NeuralPainterAgent."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_PREFIXES = ("painter/", "scripts/", "tests/")
ALLOWED_ROOT_FILES = {"README.md", "pyproject.toml"}
PROTECTED_PATHS = {
    "painter/metrics.py",
    "painter/renderer.py",
    "painter/diagnostics.py",
    "painter/research.py",
    "painter/autocode.py",
}
MAX_READ_CHARS = 24000
MAX_TOOL_OUTPUT = 16000


@dataclass(slots=True)
class ProjectTools:
    root: Path

    def _safe_relative(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            raise ValueError("absolute paths are not allowed")
        normalized = Path(*[part for part in path.parts if part not in ("", ".")])
        if ".." in normalized.parts:
            raise ValueError("parent-directory traversal is not allowed")
        posix = normalized.as_posix()
        if posix in ALLOWED_ROOT_FILES or posix.startswith(ALLOWED_PREFIXES):
            return normalized
        raise ValueError(f"path is outside the agent workspace policy: {raw}")

    def _run(self, command: list[str], *, timeout: int = 180) -> str:
        result = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        combined = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(combined[-MAX_TOOL_OUTPUT:] or "tool command failed")
        return combined[-MAX_TOOL_OUTPUT:]

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        relative = self._safe_relative(path)
        source = self.root / relative
        if not source.is_file():
            raise FileNotFoundError(path)
        lines = source.read_text(encoding="utf-8").splitlines()
        start = max(start_line - 1, 0)
        stop = len(lines) if end_line is None else min(end_line, len(lines))
        selected = "\n".join(
            f"{index + 1}: {line}"
            for index, line in enumerate(lines[start:stop], start=start)
        )
        return selected[:MAX_READ_CHARS]

    def search_code(self, query: str) -> str:
        if not query.strip():
            raise ValueError("search query must be non-empty")
        return self._run(
            [
                "rg",
                "-n",
                "--glob",
                "*.py",
                "--glob",
                "README.md",
                "--glob",
                "pyproject.toml",
                query,
                "painter",
                "scripts",
                "tests",
                "README.md",
                "pyproject.toml",
            ]
        )

    def git_diff(self) -> str:
        return self._run(
            [
                "git",
                "diff",
                "--",
                "painter",
                "scripts",
                "tests",
                "README.md",
                "pyproject.toml",
            ]
        )

    def run_ruff(self) -> str:
        return self._run(["ruff", "check", "."], timeout=300)

    def run_tests(self, target: str | None = None) -> str:
        command = ["pytest", "-q"]
        if target:
            relative = self._safe_relative(target)
            if not relative.as_posix().startswith("tests/"):
                raise ValueError("focused pytest targets must be under tests/")
            command.append(relative.as_posix())
        return self._run(command, timeout=600)

    def apply_patch(self, patch: str) -> str:
        if not patch.strip():
            raise ValueError("patch must be non-empty")
        touched = _paths_from_patch(patch)
        if not touched:
            raise ValueError("patch did not contain recognizable file paths")
        for path in touched:
            relative = self._safe_relative(path)
            posix = relative.as_posix()
            if posix in PROTECTED_PATHS:
                raise ValueError(f"agent is not allowed to modify protected file: {posix}")

        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=patch,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        return "patch applied: " + ", ".join(sorted(touched))


def _paths_from_patch(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        raw = re.sub(r"^[ab]/", "", raw)
        paths.add(raw)
    return paths
