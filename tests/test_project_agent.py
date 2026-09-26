"""Tests for the repository-native NeuralPainterAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self
from urllib.request import Request

import pytest

from painter.agent.agent import CODING_ACTION_RESPONSE_FORMAT, NeuralPainterAgent
from painter.agent.model import OpenAICompatibleModel, parse_json_action
from painter.agent.tools import ProjectTools


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        del messages, kwargs
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


def test_openai_compatible_model_requests_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "{\"action\":\"finish\"}"}}]}
            ).encode("utf-8")

    def fake_urlopen(request: Request, timeout: int) -> _Response:
        del timeout
        data = request.data
        assert data is not None
        captured.update(json.loads(data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    model = OpenAICompatibleModel("http://localhost:11434/v1", "test-model")
    result = model.complete([{"role": "user", "content": "Return JSON."}])

    assert result == '{"action":"finish"}'
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "none"
    assert captured["temperature"] == 0.0


def test_openai_compatible_model_rejects_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": ""}}]}
            ).encode("utf-8")

    def fake_urlopen(request: Request, timeout: int) -> _Response:
        del request, timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    model = OpenAICompatibleModel("http://localhost:11434/v1", "test-model")
    with pytest.raises(RuntimeError, match="empty assistant content"):
        model.complete([{"role": "user", "content": "Return JSON."}])


def test_agent_rejects_duplicate_inspection_until_source_change(tmp_path: Path) -> None:
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
            '{"action":"read_file","path":"painter/demo.py"}',
            '{"action":"finish","summary":"stopped after duplicate guard"}',
        ]
    )
    agent = NeuralPainterAgent(model, tmp_path, max_turns=4)

    result = agent.run("Inspect without looping.", log_dir=tmp_path / "logs")
    transcript = [
        json.loads(line)
        for line in result.transcript_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.success
    duplicate_observation = transcript[3]["content"]
    assert duplicate_observation["ok"] is False
    assert "duplicate inspection action" in duplicate_observation["error"]


def test_search_code_does_not_require_ripgrep(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    (tmp_path / "painter" / "demo.py").write_text(
        "def adaptive_allocator():\n    return 1\n",
        encoding="utf-8",
    )
    tools = ProjectTools(tmp_path)

    result = tools.search_code("adaptive_allocator")

    assert "painter/demo.py:1:def adaptive_allocator():" in result


def test_read_file_reports_range_and_eof_state(tmp_path: Path) -> None:
    (tmp_path / "painter").mkdir()
    (tmp_path / "painter" / "demo.py").write_text(
        "one\ntwo\nthree\n",
        encoding="utf-8",
    )
    tools = ProjectTools(tmp_path)

    partial = tools.read_file("painter/demo.py", 1, 2)
    complete = tools.read_file("painter/demo.py", 1, 3)

    assert partial.startswith("[painter/demo.py lines 1-2 of 3; more lines available]")
    assert complete.startswith("[painter/demo.py lines 1-3 of 3; EOF]")


def test_agent_requests_coding_action_schema(tmp_path: Path) -> None:
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "RESEARCH_RULES.md").write_text(
        "Implement one mutation at a time.\n",
        encoding="utf-8",
    )

    class CapturingModel:
        def __init__(self) -> None:
            self.response_format: dict[str, object] | None = None

        def complete(
            self,
            messages: list[dict[str, str]],
            **kwargs: object,
        ) -> str:
            assert messages
            value = kwargs.get("response_format")
            assert isinstance(value, dict)
            self.response_format = value
            return '{"action":"finish","summary":"done"}'

    model = CapturingModel()
    agent = NeuralPainterAgent(model, tmp_path, max_turns=1)

    result = agent.run("Finish immediately.", log_dir=tmp_path / "logs")

    assert result.success
    assert model.response_format == CODING_ACTION_RESPONSE_FORMAT
    action_schema = CODING_ACTION_RESPONSE_FORMAT["json_schema"]["schema"]
    assert "action" in action_schema["required"]
