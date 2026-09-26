"""Minimal OpenAI-compatible chat client for local/open model servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CODING_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded line range from one repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search source using a short literal identifier or phrase. "
                "This is not regex. Optionally scope to one file or directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": ["string", "null"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply one unified diff to allowed repository files.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ruff",
            "description": "Run Ruff on the repository.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest, optionally on one tests/ path.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": ["string", "null"]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Inspect the current repository diff.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish after implementing and checking the bounded mutation.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleModel:
    """Small dependency-free client for /v1/chat/completions endpoints."""

    base_url: str
    model: str
    api_key: str = ""
    timeout: int = 180

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        coding_turn = (
            isinstance(response_format, dict)
            and isinstance(response_format.get("json_schema"), dict)
            and response_format["json_schema"].get("name") == "coding_action"
        )
        if coding_turn:
            endpoint = self.base_url.removesuffix("/v1").rstrip("/") + "/api/chat"
            request_payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
                "think": False,
                "tools": CODING_TOOLS,
            }
        else:
            endpoint = self.base_url.rstrip("/") + "/chat/completions"
            request_payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
                "reasoning_effort": "none",
                "response_format": response_format or {"type": "json_object"},
            }
        payload = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"model endpoint failed at {endpoint}: {exc}"
            ) from exc

        try:
            if coding_turn:
                message = body["message"]
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    content = str(message.get("content", "")).strip()
                    if content:
                        try:
                            fallback = parse_json_action(content)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            fallback = None
                        if fallback is not None and fallback.get("action") == "finish":
                            return json.dumps(fallback)
                    return json.dumps(
                        {
                            "action": "invalid_model_response",
                            "content": content,
                        }
                    )
                function = tool_calls[0]["function"]
                name = str(function["name"])
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be an object")
                return json.dumps(
                    {
                        "action": name,
                        **arguments,
                        "_tool_name": name,
                    }
                )

            message = body["choices"][0]["message"]
            content = str(message["content"])
            if not content.strip():
                raise RuntimeError("model endpoint returned empty assistant content")
            return content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "model endpoint returned an unsupported chat-completions response"
            ) from exc


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, tolerating fenced model output."""
    stripped = text.strip()
    if stripped.startswith(("~~~", "```")):
        lines = stripped.splitlines()
        if lines and lines[0].startswith(("~~~", "```")):
            lines = lines[1:]
        if lines and lines[-1].strip() in {"~~~", "```"}:
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model did not return a JSON object")
        value = json.loads(stripped[start : end + 1])

    if not isinstance(value, dict):
        raise TypeError("model output must be a JSON object")
    return value


def parse_json_action(text: str) -> dict[str, Any]:
    """Parse a single tool action from model output."""
    value = parse_json_object(text)
    if not isinstance(value.get("action"), str):
        raise TypeError("model action must contain a string 'action'")
    return value
