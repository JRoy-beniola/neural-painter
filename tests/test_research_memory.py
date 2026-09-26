"""Tests for structured scientific memory and preregistration."""

from __future__ import annotations

import json

import pytest

from painter.agent.agent import NeuralPainterAgent
from painter.agent.memory import ResearchMemory
from painter.agent.protocol import ExperimentProtocol, evaluate_prediction


class _FakeModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, messages: list[dict[str, str]]) -> str:
        assert messages
        return json.dumps(self.payload)


def _protocol(*, direction: str = "lower", epsilon: float = 0.01) -> ExperimentProtocol:
    return ExperimentProtocol(
        id="E-001",
        question_id="Q-001",
        hypothesis_id="H-001",
        question="Does freezing cleanup geometry preserve structure?",
        hypothesis="Cleanup geometry moves already-correct structural strokes.",
        prediction="Freezing cleanup geometry should lower reconstruction MSE.",
        falsifier="MSE is unchanged or worse when cleanup geometry is frozen.",
        intervention={
            "area": "painter/refine.py",
            "change": "freeze geometry during cleanup refinement",
        },
        controls=("same image", "same seed", "same stroke budget"),
        primary_metric="mse",
        expected_direction=direction,  # type: ignore[arg-type]
        min_effect_fraction=epsilon,
        locked_at="2026-09-26T08:00:00+00:00",
    )


def test_prediction_evaluation_respects_locked_direction_and_threshold() -> None:
    protocol = _protocol(direction="lower", epsilon=0.01)

    assert evaluate_prediction(protocol, {"mse": 1.0}, {"mse": 0.98}) == "supports"
    assert evaluate_prediction(protocol, {"mse": 1.0}, {"mse": 1.02}) == "weakens"
    assert evaluate_prediction(protocol, {"mse": 1.0}, {"mse": 0.995}) == "inconclusive"


def test_research_memory_persists_evidence_and_updates_hypothesis(tmp_path) -> None:
    memory = ResearchMemory(tmp_path / "scientific_state")
    question = memory.create_question("Why does cleanup underperform?")
    hypothesis = memory.create_hypothesis(
        question_id=question.id,
        claim="Cleanup geometry damages structural placement.",
        prediction="Freezing cleanup geometry lowers MSE.",
        falsifier="Frozen cleanup geometry does not lower MSE.",
    )
    protocol = ExperimentProtocol(
        **{
            **_protocol().to_dict(),
            "question_id": question.id,
            "hypothesis_id": hypothesis.id,
        }
    )
    memory.lock_protocol(protocol)

    observation = memory.record_observation(
        experiment_id=protocol.id,
        hypothesis_id=hypothesis.id,
        before={"mse": 0.02},
        after={"mse": 0.018},
        relation="supports",
        accepted=True,
        reasons=(),
    )
    memory.record_finding(
        hypothesis_id=hypothesis.id,
        observation_id=observation.id,
        relation="supports",
        statement="Frozen cleanup geometry improved the preregistered metric.",
    )

    hypotheses = json.loads(
        (memory.root / "hypotheses.json").read_text(encoding="utf-8")
    )
    assert hypotheses[0]["status"] == "supported"
    assert memory.snapshot()["recent_findings"][0]["relation"] == "supports"
    assert (memory.root / "observations.jsonl").read_text(encoding="utf-8").strip()


def test_agent_planner_returns_validated_experiment_draft(tmp_path) -> None:
    payload = {
        "question": "Why does cleanup underperform?",
        "hypothesis": "Cleanup geometry damages structural placement.",
        "prediction": "Freezing cleanup geometry lowers MSE.",
        "falsifier": "MSE fails to improve.",
        "intervention": {
            "area": "painter/refine.py",
            "change": "freeze cleanup geometry",
        },
        "controls": ["same image", "same seed", "same budget"],
        "primary_metric": "mse",
        "expected_direction": "lower",
        "min_effect_fraction": 0.001,
    }
    agent = NeuralPainterAgent(_FakeModel(payload), tmp_path)

    draft = agent.propose_experiment(
        {
            "area": "painter/refine.py",
            "hypothesis": "cleanup may damage structure",
            "change": "separate cleanup geometry controls",
        },
        baseline_summary={
            "champion": {
                "candidate": {"method": "region_rich_refined_schedule"},
                "metrics": {"mse": 0.02, "ssim": 0.8, "diagnostics": {}},
            },
            "final_diagnoses": [],
        },
        research_state={},
    )

    assert draft.primary_metric == "mse"
    assert draft.expected_direction == "lower"
    assert draft.intervention["change"] == "freeze cleanup geometry"


def test_protocol_rejects_unknown_primary_metric() -> None:
    protocol = ExperimentProtocol(
        **{**_protocol().to_dict(), "primary_metric": "made_up_score"}
    )

    with pytest.raises(ValueError, match="unsupported primary metric"):
        protocol.validate()
