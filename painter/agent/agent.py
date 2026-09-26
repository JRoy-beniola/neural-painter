"""Neural Painter's project-specific autonomous coding loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from painter.agent.model import OpenAICompatibleModel, parse_json_action, parse_json_object
from painter.agent.protocol import ExperimentDraft, parse_experiment_draft
from painter.agent.tools import ProjectTools

CODING_ACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "coding_action",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "read_file",
                        "search_code",
                        "apply_patch",
                        "run_ruff",
                        "run_tests",
                        "git_diff",
                        "finish",
                    ],
                },
                "path": {"type": ["string", "null"]},
                "start_line": {"type": ["integer", "null"], "minimum": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
                "query": {"type": ["string", "null"]},
                "patch": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
                "summary": {"type": ["string", "null"]},
            },
            "required": [
                "action",
                "path",
                "start_line",
                "end_line",
                "query",
                "patch",
                "target",
                "summary",
            ],
        },
    },
}


EXPERIMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "experiment_draft",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "hypothesis": {"type": "string", "minLength": 1},
                "prediction": {"type": "string", "minLength": 1},
                "falsifier": {"type": "string", "minLength": 1},
                "intervention": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "area": {"type": "string", "minLength": 1},
                        "change": {"type": "string", "minLength": 1},
                    },
                    "required": ["area", "change"],
                },
                "controls": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "primary_metric": {
                    "type": "string",
                    "enum": [
                        "mse",
                        "ssim",
                        "boundary_f1",
                        "boundary_distance",
                        "high_frequency_ratio",
                        "high_frequency_boundary_ratio",
                        "high_frequency_interior_ratio",
                        "high_frequency_exterior_ratio",
                        "runtime_ms",
                    ],
                },
                "expected_direction": {
                    "type": "string",
                    "enum": ["lower", "higher", "toward_target"],
                },
                "min_effect_fraction": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "target_value": {
                    "type": ["number", "null"],
                },
            },
            "required": [
                "question",
                "hypothesis",
                "prediction",
                "falsifier",
                "intervention",
                "controls",
                "primary_metric",
                "expected_direction",
                "min_effect_fraction",
                "target_value",
            ],
        },
    },
}


EXPERIMENT_PLANNER_PROMPT = """You are the research-planning role of NeuralPainterAgent.

Turn one current diagnostic lead into exactly one falsifiable, controlled experiment.
Return one JSON object and nothing else with this schema:
{
  "question": "one explicit unresolved question",
  "hypothesis": "one mechanistic falsifiable claim",
  "prediction": "what measurable change should occur if the claim is right",
  "falsifier": "what result would count against the claim",
  "intervention": {"area": "repo/path.py", "change": "one bounded change"},
  "controls": ["what must stay fixed"],
  "primary_metric": "mse|ssim|boundary_f1|boundary_distance|high_frequency_ratio|high_frequency_boundary_ratio|high_frequency_interior_ratio|high_frequency_exterior_ratio|runtime_ms",
  "expected_direction": "lower|higher|toward_target",
  "min_effect_fraction": 0.001,
  "target_value": "number for toward_target, otherwise null"
}

Do not propose multiple simultaneous mechanisms. The intervention must remain inside the
provided diagnostic lead and repository scope. Choose a primary metric that directly tests the prediction rather than whichever metric
is easiest to improve. For any high_frequency_*_ratio metric, use
expected_direction="toward_target" and target_value=1.0. For all other metrics,
target_value must be null.
"""


SYSTEM_PROMPT = """You are NeuralPainterAgent, a narrowly scoped research coding agent.

Your entire world is the Neural Painter repository and one falsifiable mutation request.
You are not a general shell agent.

You may use only these JSON actions:
{"action":"read_file","path":"painter/x.py","start_line":1,"end_line":200}
{"action":"search_code","query":"symbol or text"}
{"action":"apply_patch","patch":"<unified git diff>"}
{"action":"run_ruff"}
{"action":"run_tests","target":"tests/test_x.py"}
{"action":"git_diff"}
{"action":"finish","summary":"what changed and why"}

Rules:
- Return exactly one JSON object per turn. No markdown outside JSON.
- Implement exactly one research mutation.
- Read before editing.
- When locating a named mechanism or function, prefer search_code over repeatedly scanning
  the same file ranges.
- Never repeat an identical read_file or search_code action unless a source patch has
  changed the repository since that inspection.
- Prefer the smallest discriminating implementation.
- Never weaken tests merely to make a change pass.
- Never edit generated outputs.
- Renderer, metrics, diagnostics, controller, and autonomous harness are protected.
- Add focused tests for new behavior when practical.
- Do not commit; the outer research harness owns git history.
- If an approach is infeasible under these constraints, finish and say so.
"""


@dataclass(frozen=True, slots=True)
class AgentResult:
    success: bool
    summary: str
    turns: int
    transcript_path: Path


class NeuralPainterAgent:
    """Tool-constrained coding agent driven by a swappable local/open model."""

    def __init__(
        self,
        model: OpenAICompatibleModel,
        worktree: Path,
        *,
        max_turns: int = 24,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.model = model
        self.tools = ProjectTools(worktree)
        self.max_turns = max_turns

    def propose_experiment(
        self,
        mutation: dict[str, str],
        *,
        baseline_summary: dict[str, Any],
        research_state: dict[str, Any],
    ) -> ExperimentDraft:
        """Formulate one schema-validated preregisterable experiment."""
        champion = baseline_summary.get("champion") or {}
        metrics = champion.get("metrics") or {}
        payload = {
            "diagnostic_lead": mutation,
            "current_champion": {
                "method": champion.get("candidate", {}).get("method"),
                "mse": metrics.get("mse"),
                "ssim": metrics.get("ssim"),
                "diagnostics": metrics.get("diagnostics", {}),
            },
            "active_diagnoses": baseline_summary.get("final_diagnoses") or [],
            "research_state": research_state,
        }
        raw = self.model.complete(
            [
                {"role": "system", "content": EXPERIMENT_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
            response_format=EXPERIMENT_RESPONSE_FORMAT,
        )
        draft = parse_experiment_draft(parse_json_object(raw))
        expected_area = mutation.get("area", "").strip()
        if expected_area and draft.intervention["area"] != expected_area:
            raise ValueError(
                "research planner may not broaden the controller-provided code area"
            )
        return draft

    def run(self, task: str, *, log_dir: Path) -> AgentResult:
        log_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = log_dir / "agent_transcript.jsonl"
        rules_path = self.tools.root / ".agent" / "RESEARCH_RULES.md"
        project_rules = (
            rules_path.read_text(encoding="utf-8")
            if rules_path.is_file()
            else "No additional project rules file was found."
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nPROJECT RESEARCH RULES:\n" + project_rules,
            },
            {"role": "user", "content": task},
        ]

        inspection_keys: set[str] = set()
        source_changed = False

        with transcript_path.open("w", encoding="utf-8") as transcript:
            for turn in range(1, self.max_turns + 1):
                raw = self.model.complete(
                    messages,
                    response_format=CODING_ACTION_RESPONSE_FORMAT,
                )
                transcript.write(
                    json.dumps({"turn": turn, "kind": "model", "content": raw}) + "\n"
                )
                transcript.flush()

                tool_call_id: str | None = None
                try:
                    action = parse_json_action(raw)
                    tool_call_id = action.pop("_tool_call_id", None)
                    name = action.get("action")
                    inspection_key = (
                        json.dumps(action, sort_keys=True, separators=(",", ":"))
                        if name in {"read_file", "search_code"}
                        else None
                    )
                    if inspection_key is not None and inspection_key in inspection_keys:
                        observation = {
                            "ok": False,
                            "error": (
                                "duplicate inspection action: this exact read/search already "
                                "succeeded since the last source change. Do not repeat it."
                            ),
                            "progress_required": (
                                "Search for a different symbol or range, apply the bounded "
                                "patch, run focused tests, inspect git_diff, or finish."
                            ),
                        }
                    else:
                        if inspection_key is not None:
                            inspection_keys.add(inspection_key)
                        observation = self._execute(action)
                        if observation.get("ok") and name == "apply_patch":
                            source_changed = True
                            inspection_keys.clear()
                except Exception as exc:  # noqa: BLE001
                    action = {"action": "invalid"}
                    observation = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                transcript.write(
                    json.dumps(
                        {"turn": turn, "kind": "observation", "content": observation}
                    )
                    + "\n"
                )
                transcript.flush()

                if action.get("action") == "finish" and observation.get("ok"):
                    return AgentResult(
                        success=True,
                        summary=str(action.get("summary", "")),
                        turns=turn,
                        transcript_path=transcript_path,
                    )

                progress_state = {
                    "source_changed": source_changed,
                    "unique_inspections_since_change": len(inspection_keys),
                    "turns_remaining": self.max_turns - turn,
                }
                if tool_call_id is not None and action.get("action") != "invalid":
                    arguments = {
                        key: value
                        for key, value in action.items()
                        if key != "action"
                    }
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": str(tool_call_id),
                                    "type": "function",
                                    "function": {
                                        "name": str(action["action"]),
                                        "arguments": json.dumps(
                                            arguments,
                                            ensure_ascii=False,
                                        ),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call_id),
                            "content": json.dumps(
                                {
                                    "observation": observation,
                                    "progress_state": progress_state,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "TOOL OBSERVATION:\n"
                                + json.dumps(observation, ensure_ascii=False)
                                + "\nPROGRESS STATE:\n"
                                + json.dumps(progress_state)
                            ),
                        }
                    )

        return AgentResult(
            success=False,
            summary="agent exceeded its turn budget",
            turns=self.max_turns,
            transcript_path=transcript_path,
        )

    def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        name = action["action"]
        if name == "read_file":
            output = self.tools.read_file(
                str(action["path"]),
                int(action.get("start_line", 1)),
                int(action["end_line"]) if action.get("end_line") is not None else None,
            )
        elif name == "search_code":
            output = self.tools.search_code(str(action["query"]))
        elif name == "apply_patch":
            output = self.tools.apply_patch(str(action["patch"]))
        elif name == "run_ruff":
            output = self.tools.run_ruff()
        elif name == "run_tests":
            target = action.get("target")
            output = self.tools.run_tests(str(target) if target else None)
        elif name == "git_diff":
            output = self.tools.git_diff()
        elif name == "finish":
            return {"ok": True, "summary": str(action.get("summary", ""))}
        else:
            raise ValueError(f"unsupported action: {name}")
        return {"ok": True, "output": output}
