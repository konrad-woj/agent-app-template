"""Pydantic models for evals/ — Phase 5d (see PLAN.md).

`ExpectedOutput` is the schema every item's `expected_output` in
evals/datasets/*.yaml must satisfy. It is validated at dataset-load time (see
`run_experiment.py:_validate_expected_outputs`), not lazily inside an
evaluator — a misspelled field in a hand-edited YAML file must raise loudly
before a run starts, not silently make one evaluator a permanent no-op.

`RubricJudgeVerdict` is a **provisional** structural model for the current
(pre-Phase-5b) freeform-JSON quality judge output — see evals/evaluators.py.
Phase 5b will replace it with per-criterion Pydantic models
(`CriterionVerdict`/`HolisticCriterionVerdict`) enforcing a `reasoning` field
written before `score`, plus a `confidence` flag. This model exists only so
`run_experiment.py` has *something* to validate the judge's JSON against in
the meantime, rather than trusting `json.loads` blindly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PainPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str


class ExpectedOutput(BaseModel):
    """Matches the expected_output shape used across evals/datasets/sample.yaml.

    `extra="forbid"` is deliberate: a typo'd field (e.g. `must_include_term`
    instead of `must_include_terms`) must raise at load time instead of
    silently validating as an empty-defaulted, differently-named field.
    """

    model_config = ConfigDict(extra="forbid")

    must_address: list[PainPoint] = []
    must_reference: list[str] = []
    must_include_terms: list[str] = []
    must_not_contain: list[str] = []
    scoring_hints: list[str] = []


class CriterionScore(BaseModel):
    """One criterion's score within a RubricJudgeVerdict — see module docstring."""

    model_config = ConfigDict(extra="forbid")

    score: int
    reason: str


class RubricJudgeVerdict(BaseModel):
    """Structural validation for the current freeform quality-judge JSON output.

    Matches the shape documented in evals/configs/scoring_rubric.yaml's
    `pass_policy` comment: {"scores": {...}, "total": int, "passed": bool,
    "summary": str}. See module docstring for the Phase 5b upgrade path.
    """

    model_config = ConfigDict(extra="forbid")

    scores: dict[str, CriterionScore]
    total: int
    passed: bool
    summary: str
