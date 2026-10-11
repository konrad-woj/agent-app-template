"""Per-criterion LLM-judge verdict models shared by the reflection critic and evals/.

Two design rules, both from PLAN.md Phase 5b (rationale and citations live there):

  1. Field order is reasoning → score → confidence → reason. Pydantic preserves
     declaration order in the JSON schema, so a judge filling the object top to
     bottom writes its evidence before it commits to a number.
  2. Coarse scales, picked per criterion: 0–4 when evidence is enumerable (you
     can point at sentences and count), 0–2 when the criterion is inherently
     holistic. Never binary, never continuous.

`confidence="low"` exists so an uncertain verdict is surfaced for human review
instead of silently averaging into a pass/fail total — callers must collect it
(see `low_confidence_criteria`), not drop it.

Unknown extra keys are ignored (Pydantic's default), not forbidden: these
models validate LLM output, and a judge adding a stray key should not turn a
usable verdict into a parse failure.
"""

from typing import Literal

from pydantic import BaseModel


class CriterionVerdict(BaseModel):
    """0–4 verdict for a criterion with checkable, enumerable evidence."""

    reasoning: str  # written first — forces evidence-gathering before scoring
    score: Literal[0, 1, 2, 3, 4]
    confidence: Literal["high", "low"]
    reason: str  # one-line summary for the report/log


class HolisticCriterionVerdict(BaseModel):
    """0–2 verdict for an inherently holistic/subjective criterion."""

    reasoning: str
    score: Literal[0, 1, 2]
    confidence: Literal["high", "low"]
    reason: str


type AnyCriterionVerdict = CriterionVerdict | HolisticCriterionVerdict


def criterion_verdicts(verdict: BaseModel) -> dict[str, AnyCriterionVerdict]:
    """Return {criterion_id: verdict} for every per-criterion field on `verdict`."""
    return {
        name: value
        for name, value in ((n, getattr(verdict, n)) for n in type(verdict).model_fields)
        if isinstance(value, (CriterionVerdict, HolisticCriterionVerdict))
    }


def max_score_of(verdict_type: type[AnyCriterionVerdict]) -> int:
    """Top of the scale a verdict model allows (4 or 2)."""
    return 4 if verdict_type is CriterionVerdict else 2


def low_confidence_criteria(verdict: BaseModel) -> list[str]:
    """Criterion ids the judge itself flagged as uncertain, in declaration order."""
    return [name for name, v in criterion_verdicts(verdict).items() if v.confidence == "low"]
