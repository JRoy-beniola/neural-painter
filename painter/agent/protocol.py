"""Pre-registration schema and deterministic prediction evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Direction = Literal["lower", "higher"]
Relation = Literal["supports", "weakens", "inconclusive"]

ALLOWED_PRIMARY_METRICS = frozenset(
    {
        "mse",
        "ssim",
        "boundary_f1",
        "boundary_distance",
        "runtime_ms",
    }
)


@dataclass(frozen=True, slots=True)
class ExperimentDraft:
    """Model-proposed experiment before project code assigns stable IDs."""

    question: str
    hypothesis: str
    prediction: str
    falsifier: str
    intervention: dict[str, str]
    controls: tuple[str, ...]
    primary_metric: str
    expected_direction: Direction
    min_effect_fraction: float = 0.001

    def validate(self) -> None:
        required = {
            "question": self.question,
            "hypothesis": self.hypothesis,
            "prediction": self.prediction,
            "falsifier": self.falsifier,
        }
        for name, value in required.items():
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.primary_metric not in ALLOWED_PRIMARY_METRICS:
            raise ValueError(f"unsupported primary metric: {self.primary_metric}")
        if self.expected_direction not in {"lower", "higher"}:
            raise ValueError("expected_direction must be 'lower' or 'higher'")
        if not 0.0 <= self.min_effect_fraction <= 1.0:
            raise ValueError("min_effect_fraction must be between 0 and 1")
        if not self.intervention.get("area", "").strip():
            raise ValueError("intervention.area must be non-empty")
        if not self.intervention.get("change", "").strip():
            raise ValueError("intervention.change must be non-empty")
        if not self.controls or any(not item.strip() for item in self.controls):
            raise ValueError("controls must contain at least one non-empty item")


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """Locked experiment specification evaluated after measurement."""

    id: str
    question_id: str
    hypothesis_id: str
    question: str
    hypothesis: str
    prediction: str
    falsifier: str
    intervention: dict[str, str]
    controls: tuple[str, ...]
    primary_metric: str
    expected_direction: Direction
    min_effect_fraction: float
    locked_at: str

    def validate(self) -> None:
        ExperimentDraft(
            question=self.question,
            hypothesis=self.hypothesis,
            prediction=self.prediction,
            falsifier=self.falsifier,
            intervention=self.intervention,
            controls=self.controls,
            primary_metric=self.primary_metric,
            expected_direction=self.expected_direction,
            min_effect_fraction=self.min_effect_fraction,
        ).validate()
        if not self.id.strip() or not self.question_id.strip() or not self.hypothesis_id.strip():
            raise ValueError("protocol IDs must be non-empty")
        if not self.locked_at.strip():
            raise ValueError("locked_at must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_experiment_draft(payload: dict[str, Any]) -> ExperimentDraft:
    """Convert untrusted model JSON into a validated experiment draft."""
    intervention = payload.get("intervention")
    controls = payload.get("controls")
    if not isinstance(intervention, dict):
        raise TypeError("intervention must be an object")
    if not isinstance(controls, list) or not all(isinstance(item, str) for item in controls):
        raise TypeError("controls must be a list of strings")

    draft = ExperimentDraft(
        question=str(payload.get("question", "")),
        hypothesis=str(payload.get("hypothesis", "")),
        prediction=str(payload.get("prediction", "")),
        falsifier=str(payload.get("falsifier", "")),
        intervention={
            "area": str(intervention.get("area", "")),
            "change": str(intervention.get("change", "")),
        },
        controls=tuple(controls),
        primary_metric=str(payload.get("primary_metric", "")),
        expected_direction=str(payload.get("expected_direction", "")),  # type: ignore[arg-type]
        min_effect_fraction=float(payload.get("min_effect_fraction", 0.001)),
    )
    draft.validate()
    return draft


def evaluate_prediction(
    protocol: ExperimentProtocol,
    before: dict[str, float],
    after: dict[str, float],
) -> Relation:
    """Judge only the pre-registered primary prediction, independent of acceptance."""
    protocol.validate()
    metric = protocol.primary_metric
    if metric not in before or metric not in after:
        raise KeyError(f"missing primary metric {metric!r} from measured comparison")

    old = float(before[metric])
    new = float(after[metric])
    scale = max(abs(old), 1e-12)
    fractional_change = (new - old) / scale
    epsilon = protocol.min_effect_fraction

    if protocol.expected_direction == "lower":
        if fractional_change <= -epsilon:
            return "supports"
        if fractional_change >= epsilon:
            return "weakens"
    else:
        if fractional_change >= epsilon:
            return "supports"
        if fractional_change <= -epsilon:
            return "weakens"
    return "inconclusive"
