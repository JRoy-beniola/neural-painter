"""Tests for the bounded one-shot Neural Painter code loop."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from painter.code_loop import (
    build_mutation_prompt,
    extract_function_context,
    parse_unified_diff,
    validate_target_patch,
)


def test_extract_function_context_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    path = tmp_path / "painter" / "demo.py"
    path.write_text(
        "VALUE = 1\n\n"
        "def target(x: int) -> int:\n"
        "    return x + 1\n\n"
        "OTHER = 2\n",
        encoding="utf-8",
    )

    target = extract_function_context(
        tmp_path, "painter/demo.py", "target", context_lines=1
    )

    assert target.function_name == "target"
    assert "def target" in target.source
    assert "return x + 1" in target.source


def test_prompt_locks_model_to_one_file(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    (tmp_path / "painter" / "demo.py").write_text(
        "def target() -> int:\n    return 1\n", encoding="utf-8"
    )
    target = extract_function_context(tmp_path, "painter/demo.py", "target")

    prompt = build_mutation_prompt(
        target, objective="Clarify one comment.", behavior_preserving=True
    )

    assert "Modify only painter/demo.py" in prompt
    assert "preserve executable behavior" in prompt
    assert "Do not use Markdown fences" in prompt


def test_parse_unified_diff_rejects_prose() -> None:
    with pytest.raises(ValueError, match="prose before"):
        parse_unified_diff(
            "Here is the patch:\n"
            "--- a/painter/demo.py\n"
            "+++ b/painter/demo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )


def test_validate_target_patch_requires_exact_file(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    (tmp_path / "painter" / "demo.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "painter" / "other.py").write_text("old\n", encoding="utf-8")
    init = subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert init.returncode == 0

    patch = (
        "--- a/painter/other.py\n"
        "+++ b/painter/other.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    with pytest.raises(ValueError, match="must touch only painter/demo.py"):
        validate_target_patch(tmp_path, patch, target_path="painter/demo.py")
