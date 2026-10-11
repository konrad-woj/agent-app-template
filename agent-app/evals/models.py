"""Pydantic models for evals/ — Phase 5d (see PLAN.md).

`ExpectedOutput` is the schema every item's `expected_output` in
evals/datasets/*.yaml must satisfy. It is validated at dataset-load time (see
`run_experiment.py:_validate_expected_outputs`), not lazily inside an
evaluator — a misspelled field in a hand-edited YAML file must raise loudly
before a run starts, not silently make one evaluator a permanent no-op.

`QualityVerdict` is the quality judge's output schema (Phase 5b): one
reasoning → score → confidence verdict per `quality_criteria` entry in
evals/configs/scoring_rubric.yaml, built from the same per-criterion models the
reflection critic uses (app/graph/nodes/subgraphs/_rubric_models.py). The judge
is not asked for a total or a pass/fail — evals/evaluators.py computes both
from the scores and the rubric's pass_policy, so judge arithmetic can't drift
from the rubric.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.graph.nodes.subgraphs._rubric_models import CriterionVerdict, HolisticCriterionVerdict


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


class QualityVerdict(BaseModel):
    """One verdict per scoring_rubric.yaml criterion — field names are the criterion ids.

    0–4 (`CriterionVerdict`) for criteria with checkable evidence, 0–2
    (`HolisticCriterionVerdict`) for holistic ones; the rubric's `max_score`
    values must agree, which `evals.evaluators.check_rubric_matches_verdict`
    enforces at config-load time.
    """

    stack_alignment: CriterionVerdict
    pain_point_coverage: CriterionVerdict
    specificity: CriterionVerdict
    feasibility: HolisticCriterionVerdict
    risk_acknowledgment: HolisticCriterionVerdict
    summary: str
