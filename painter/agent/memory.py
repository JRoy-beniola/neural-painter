"""Durable, schema-managed scientific memory for Neural Painter."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from painter.agent.hypothesis import Finding, Hypothesis, Observation, ResearchQuestion
from painter.agent.protocol import ExperimentProtocol


class ResearchMemory:
    """Persist scientific state without giving the coding agent direct write access."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_json(self.root / "questions.json")
        self._ensure_json(self.root / "hypotheses.json")
        self._ensure_json(self.root / "protocols.json")
        self._ensure_json(self.root / "findings.json")
        (self.root / "observations.jsonl").touch(exist_ok=True)

    @staticmethod
    def _ensure_json(path: Path) -> None:
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")

    @staticmethod
    def _read_list(path: Path) -> list[dict[str, Any]]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"research state file must contain a list: {path}")
        return value

    @staticmethod
    def _atomic_write(path: Path, records: Iterable[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(list(records), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _next_id(records: list[dict[str, Any]], prefix: str) -> str:
        maximum = 0
        for record in records:
            raw = str(record.get("id", ""))
            if raw.startswith(prefix + "-"):
                try:
                    maximum = max(maximum, int(raw.split("-", 1)[1]))
                except ValueError:
                    continue
        return f"{prefix}-{maximum + 1:03d}"

    def create_question(self, text: str) -> ResearchQuestion:
        if not text.strip():
            raise ValueError("research question must be non-empty")
        path = self.root / "questions.json"
        records = self._read_list(path)
        question = ResearchQuestion(id=self._next_id(records, "Q"), text=text.strip())
        records.append(question.to_dict())
        self._atomic_write(path, records)
        return question

    def create_hypothesis(
        self,
        *,
        question_id: str,
        claim: str,
        prediction: str,
        falsifier: str,
        source: str = "model",
    ) -> Hypothesis:
        if not all(item.strip() for item in (question_id, claim, prediction, falsifier)):
            raise ValueError("hypothesis fields must be non-empty")
        path = self.root / "hypotheses.json"
        records = self._read_list(path)
        hypothesis = Hypothesis(
            id=self._next_id(records, "H"),
            question_id=question_id,
            claim=claim.strip(),
            prediction=prediction.strip(),
            falsifier=falsifier.strip(),
            source=source,
        )
        records.append(hypothesis.to_dict())
        self._atomic_write(path, records)
        return hypothesis

    def next_experiment_id(self) -> str:
        records = self._read_list(self.root / "protocols.json")
        return self._next_id(records, "E")

    def lock_protocol(self, protocol: ExperimentProtocol) -> None:
        protocol.validate()
        path = self.root / "protocols.json"
        records = self._read_list(path)
        if any(record.get("id") == protocol.id for record in records):
            raise ValueError(f"protocol already exists: {protocol.id}")
        records.append(protocol.to_dict())
        self._atomic_write(path, records)

    def record_observation(
        self,
        *,
        experiment_id: str,
        hypothesis_id: str,
        before: dict[str, float],
        after: dict[str, float],
        relation: str,
        accepted: bool,
        reasons: tuple[str, ...],
    ) -> Observation:
        observations = self._read_observations()
        observation = Observation(
            id=self._next_id(observations, "O"),
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            before=before,
            after=after,
            relation=relation,
            accepted=accepted,
            reasons=reasons,
        )
        with (self.root / "observations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True) + "\n")
        return observation

    def record_finding(
        self,
        *,
        hypothesis_id: str,
        observation_id: str,
        relation: str,
        statement: str,
    ) -> Finding:
        if relation not in {"supports", "weakens", "inconclusive"}:
            raise ValueError(f"unsupported evidence relation: {relation}")
        path = self.root / "findings.json"
        records = self._read_list(path)
        finding = Finding(
            id=self._next_id(records, "F"),
            hypothesis_id=hypothesis_id,
            observation_id=observation_id,
            relation=relation,
            statement=statement.strip(),
        )
        records.append(finding.to_dict())
        self._atomic_write(path, records)
        self._update_hypothesis_status(hypothesis_id, relation)
        return finding

    def record_inconclusive(
        self,
        *,
        hypothesis_id: str,
        experiment_id: str,
        reason: str,
    ) -> Finding:
        observation = self.record_observation(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            before={},
            after={},
            relation="inconclusive",
            accepted=False,
            reasons=(reason,),
        )
        return self.record_finding(
            hypothesis_id=hypothesis_id,
            observation_id=observation.id,
            relation="inconclusive",
            statement=reason,
        )

    def _read_observations(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in (self.root / "observations.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("observation log entries must be objects")
                records.append(value)
        return records

    def _update_hypothesis_status(self, hypothesis_id: str, relation: str) -> None:
        path = self.root / "hypotheses.json"
        records = self._read_list(path)
        changed = False
        for index, record in enumerate(records):
            if record.get("id") != hypothesis_id:
                continue
            hypothesis = Hypothesis(**record)
            status = {
                "supports": "supported",
                "weakens": "weakened",
                "inconclusive": "active",
            }[relation]
            records[index] = replace(hypothesis, status=status).to_dict()
            changed = True
            break
        if not changed:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")
        self._atomic_write(path, records)

    def snapshot(self, *, recent_limit: int = 8) -> dict[str, Any]:
        """Return compact state suitable for model context."""
        hypotheses = self._read_list(self.root / "hypotheses.json")
        findings = self._read_list(self.root / "findings.json")
        questions = self._read_list(self.root / "questions.json")
        observations = self._read_observations()
        return {
            "open_questions": [item for item in questions if item.get("status") == "open"][-recent_limit:],
            "active_hypotheses": [
                item for item in hypotheses if item.get("status") == "active"
            ][-recent_limit:],
            "recent_findings": findings[-recent_limit:],
            "recent_observations": observations[-recent_limit:],
        }
