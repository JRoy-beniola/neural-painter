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
            "description": "Search repository source for one short literal symbol or phrase.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
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
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        coding_turn = (
            isinstance(response_format, dict)
            and isinstance(response_format.get("json_schema"), dict)
            and response_format["json_schema"].get("name") == "coding_action"
        )
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
            "reasoning_effort": "none",
        }
        if coding_turn:
            request_payload["tools"] = CODING_TOOLS
            request_payload["tool_choice"] = "required"
        else:
            request_payload["response_format"] = response_format or {
                "type": "json_object"
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
            message = body["choices"][0]["message"]
            if coding_turn:
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    raise RuntimeError("model endpoint returned no coding tool call")
                function = tool_calls[0]["function"]
                name = str(function["name"])
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool-call arguments must be an object")
                call_id = str(tool_calls[0].get("id") or f"call_{name}")
                return json.dumps(
                    {
                        "action": name,
                        **arguments,
                        "_tool_call_id": call_id,
                    }
                )

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
