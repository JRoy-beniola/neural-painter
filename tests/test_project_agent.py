"""Tests for the repository-native NeuralPainterAgent."""

from __future__ import annotations

from pathlib import Path

import pytest

from painter.agent.agent import NeuralPainterAgent
from painter.agent.model import parse_json_action
from painter.agent.tools import ProjectTools


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        del messages
        if not self.responses:
            raise RuntimeError("no fake response left")
        return self.responses.pop(0)


def test_parse_json_action_accepts_fenced_json() -> None:
    action = parse_json_action(
        """```json
{"action":"finish","summary":"done"}
```"""
    )
    assert action["action"] == "finish"
    assert action["summary"] == "done"


def test_project_tools_reject_parent_traversal(tmp_path: Path) -> None:
    tools = ProjectTools(tmp_path)
    with pytest.raises(ValueError):
        tools.read_file("../secret.txt")


def test_project_tools_reject_protected_patch(tmp_path: Path) -> None:
    tools = ProjectTools(tmp_path)
    patch = """diff --git a/painter/research.py b/painter/research.py
--- a/painter/research.py
+++ b/painter/research.py
@@ -1 +1 @@
-old
+new
"""
    with pytest.raises(ValueError, match="protected"):
        tools.apply_patch(patch)


def test_agent_can_read_then_finish(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    (tmp_path / "painter" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "RESEARCH_RULES.md").write_text(
        "Implement one mutation at a time.\n",
        encoding="utf-8",
    )

    model = FakeModel(
        [
            '{"action":"read_file","path":"painter/demo.py"}',
            '{"action":"finish","summary":"inspected repository state"}',
        ]
    )
    agent = NeuralPainterAgent(model, tmp_path, max_turns=4)

    result = agent.run("Inspect demo.py.", log_dir=tmp_path / "logs")

    assert result.success
    assert result.turns == 2
    assert result.transcript_path.exists()
