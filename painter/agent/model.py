"""Minimal OpenAI-compatible chat client for local/open model servers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OpenAICompatibleModel:
    """Small dependency-free client for /v1/chat/completions endpoints."""

    base_url: str
    model: str
    api_key: str = ""
    timeout: int = 180

    def complete(self, messages: list[dict[str, str]]) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "stream": False,
            }
        ).encode("utf-8")
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
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
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
            raise ValueError("model did not return a JSON action")
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
