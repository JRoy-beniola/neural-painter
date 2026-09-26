"""Structured scientific objects for Neural Painter autoresearch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    """One explicit unresolved research question."""

    id: str
    text: str
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A falsifiable claim tied to one research question."""

    id: str
    question_id: str
    claim: str
    prediction: str
    falsifier: str
    status: str = "active"
    source: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Observation:
    """Measured evidence from one locked experiment."""

    id: str
    experiment_id: str
    hypothesis_id: str
    before: dict[str, float]
    after: dict[str, float]
    relation: str
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Finding:
    """A durable interpretation of an observation."""

    id: str
    hypothesis_id: str
    observation_id: str
    relation: str
    statement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
