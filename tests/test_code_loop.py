"""Tests for the bounded one-shot Neural Painter code loop."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from painter.code_loop import (
    build_function_patch,
    build_mutation_prompt,
    extract_function_context,
    parse_replacement_function,
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
    assert target.function_source == "def target(x: int) -> int:\n    return x + 1\n"


def test_prompt_requests_replacement_function(tmp_path: Path) -> None:
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
    assert "complete replacement definition of target" in prompt
    assert "no unified diff" in prompt


def test_parse_replacement_function_rejects_prose() -> None:
    original = "def target() -> int:\n    return 1\n"
    with pytest.raises(ValueError, match="prose before"):
        parse_replacement_function(
            "Here is the function:\ndef target() -> int:\n    return 1\n",
            function_name="target",
            original_source=original,
        )


def test_parse_replacement_function_rejects_signature_change() -> None:
    original = "def target() -> int:\n    return 1\n"
    with pytest.raises(ValueError, match="signature"):
        parse_replacement_function(
            "def target(value: int) -> int:\n    return value\n",
            function_name="target",
            original_source=original,
        )


def test_build_function_patch_is_git_applicable(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    path = tmp_path / "painter" / "demo.py"
    path.write_text(
        "VALUE = 1\n\ndef target() -> int:\n    return 1\n\nOTHER = 2\n",
        encoding="utf-8",
    )
    init = subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert init.returncode == 0
    target = extract_function_context(tmp_path, "painter/demo.py", "target")
    patch = build_function_patch(
        tmp_path, target, "def target() -> int:\n    \"\"\"Clearer.\"\"\"\n    return 1\n"
    )

    validate_target_patch(tmp_path, patch, target_path="painter/demo.py")
    assert "--- a/painter/demo.py" in patch
    assert "+++ b/painter/demo.py" in patch


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
