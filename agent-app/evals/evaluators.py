"""Evaluators for evals/run_experiment.py — Phases 5d and 7 (see PLAN.md).

Five evaluators, each returning a Langfuse `Evaluation` so they plug directly
into `create_score()` when Langfuse is configured (see run_experiment.py):

  quality_score_evaluator    — LLM judge scores final_answer against
                               quality_criteria (evals/configs/scoring_rubric.yaml)
                               as a QualityVerdict, using the item's must_cover key
                               points and scoring_hints as its reference; total and
                               pass/fail are computed here from the rubric, not by the judge.
  expected_terms_evaluator   — deterministic must_include_terms / must_not_contain
                               check on final_answer. No LLM call.
  trace_assertion_evaluator  — deterministic checks against checkpoint history +
                               final state (evals/trace_assertions.py). No LLM call.
  turns_to_complete_evaluator — how many /v1/chat turns the run took.
  plan_approved_evaluator    — the planner interrupt fired AND the resume
                               decision matches what the dataset item asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Evaluation
from logger import get_logger

from app.exceptions import LLMError
from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm, llm_invoke_with_retry, parse_structured
from app.graph.nodes.subgraphs._rubric_models import criterion_verdicts, low_confidence_criteria, max_score_of
from evals.models import ExpectedOutput, QualityVerdict
from evals.trace_assertions import evaluate_trace_assertions

logger = get_logger(__name__)

_JUDGE_CONFIG: RunnableConfig = {"configurable": {"thread_id": "quality-judge"}}


def turns_to_complete_evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
    _ = expected_output, kwargs
    turns = (output or {}).get("turns")
    if turns is None:
        return Evaluation(name="turns_to_complete", value=0.0, comment="no turns recorded", data_type="NUMERIC")
    return Evaluation(name="turns_to_complete", value=float(turns), data_type="NUMERIC")


def plan_approved_evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
    """Passes when the interrupt fired AND the final status matches the requested approval.

    Not just "did an interrupt happen" — that alone doesn't prove the resume flow
    respected the caller's approve/reject decision, which is the behaviour actually
    worth regression-testing here.
    """
    _ = expected_output, kwargs
    out = output or {}
    if not out.get("reached_interrupt"):
        return Evaluation(
            name="plan_approved", value=0, comment="run never reached the planner interrupt", data_type="BOOLEAN"
        )
    approve_plan = out.get("approve_plan")
    final_status = out.get("status")
    ok = final_status != "aborted" if approve_plan else final_status == "aborted"
    return Evaluation(
        name="plan_approved",
        value=int(ok),
        comment=f"approve_plan={approve_plan}, final status={final_status!r}",
        data_type="BOOLEAN",
    )


def expected_terms_evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
    """Fraction of term checks final_answer passes: each required term present, each forbidden phrase absent.

    Case-insensitive substring match — a cheap regression signal next to the
    LLM judge, not a quality score on its own.
    """
    _ = kwargs
    final_answer = (output or {}).get("final_answer")
    if not final_answer:
        return Evaluation(name="expected_terms", value=0.0, comment="no final_answer produced", data_type="NUMERIC")
    expected = ExpectedOutput.model_validate(expected_output or {})
    answer = final_answer.lower()
    missing = [term for term in expected.must_include_terms if term.lower() not in answer]
    forbidden = [phrase for phrase in expected.must_not_contain if phrase.lower() in answer]
    check_count = len(expected.must_include_terms) + len(expected.must_not_contain)
    if not check_count:
        return Evaluation(name="expected_terms", value=1.0, comment="no term checks defined", data_type="NUMERIC")
    parts = []
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if forbidden:
        parts.append("forbidden found: " + ", ".join(forbidden))
    return Evaluation(
        name="expected_terms",
        value=(check_count - len(missing) - len(forbidden)) / check_count,
        comment="; ".join(parts) or "all term checks passed",
        data_type="NUMERIC",
    )


def make_trace_assertion_evaluator(assertions: list[dict]) -> Callable:
    """Return an evaluator closing over `assertions`, loaded once at config-parse time."""

    def evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
        _ = expected_output, kwargs
        out = output or {}
        history = out.get("history", [])
        final_state = out.get("final_state", {})
        if not history:
            return Evaluation(
                name="trace_assertions", value=0.0, comment="no checkpoint history available", data_type="NUMERIC"
            )
        results = evaluate_trace_assertions(assertions, history, final_state)
        if not results:
            return Evaluation(
                name="trace_assertions", value=1.0, comment="no applicable assertions", data_type="NUMERIC"
            )
        failed = [r for r in results if not r.passed]
        pass_rate = (len(results) - len(failed)) / len(results)
        comment = "; ".join(f"{r.id}: {r.detail}" for r in failed) or "all assertions passed"
        return Evaluation(name="trace_assertions", value=pass_rate, comment=comment, data_type="NUMERIC")

    return evaluator


def check_rubric_matches_verdict(criteria: list[dict]) -> None:
    """Fail loudly if scoring_rubric.yaml and QualityVerdict disagree.

    The criterion ids and each max_score must match QualityVerdict's fields and
    their scale (4 or 2). Called at config-load time so a hand-edited rubric
    can't silently score against a schema it no longer describes.

    Raises:
        ValueError: listing every mismatch.
    """
    expected = {
        name: max_score_of(field.annotation)  # type: ignore[arg-type]
        for name, field in QualityVerdict.model_fields.items()
        if name != "summary"
    }
    actual = {c["id"]: c["max_score"] for c in criteria}
    problems = [f"missing from rubric: {k}" for k in expected.keys() - actual.keys()]
    problems += [f"not in QualityVerdict: {k}" for k in actual.keys() - expected.keys()]
    problems += [
        f"{k}: rubric max_score={actual[k]}, QualityVerdict scale is 0-{expected[k]}"
        for k in expected.keys() & actual.keys()
        if actual[k] != expected[k]
    ]
    if problems:
        raise ValueError(
            "scoring_rubric.yaml does not match evals.models.QualityVerdict: " + "; ".join(sorted(problems))
        )


@dataclass(frozen=True)
class RubricScore:
    total: int
    max_total: int
    passed: bool
    below_minimum: list[str]  # criterion ids under their per_criterion_minimum
    low_confidence: list[str]  # criterion ids the judge flagged — needs a human read


def score_verdict(verdict: QualityVerdict, criteria: list[dict], minimum_total: int) -> RubricScore:
    """Apply the rubric's pass_policy to a verdict. Pure — no LLM call."""
    scores = {name: v.score for name, v in criterion_verdicts(verdict).items()}
    total = sum(scores.values())
    below = [c["id"] for c in criteria if scores[c["id"]] < c.get("per_criterion_minimum", 0)]
    return RubricScore(
        total=total,
        max_total=sum(c["max_score"] for c in criteria),
        passed=not below and total >= minimum_total,
        below_minimum=below,
        low_confidence=low_confidence_criteria(verdict),
    )


def _build_reference_block(expected: ExpectedOutput | None) -> str:
    if expected is None:
        return ""
    sections = []
    if expected.must_cover:
        points = "\n".join(f"- {point.description}" for point in expected.must_cover)
        sections.append(f"Key points a complete answer covers (reference for key_point_coverage):\n{points}")
    if expected.scoring_hints:
        hints = "\n".join(f"- {hint}" for hint in expected.scoring_hints)
        sections.append(f"Scoring hints:\n{hints}")
    return "".join(f"{section}\n\n" for section in sections)


def _build_rubric_prompt(
    question: str, final_answer: str, criteria: list[dict], expected: ExpectedOutput | None = None
) -> str:
    criteria_text = "\n\n".join(
        f"- {c['id']} ({c['name']}, score 0-{c['max_score']}): {c['description'].strip()}\n"
        + "\n".join(f"    {score}: {desc}" for score, desc in c["scoring_guide"].items())
        for c in criteria
    )
    shape = ", ".join(
        f'"{c["id"]}": {{"reasoning": "...", "score": 0-{c["max_score"]}, "confidence": "high" | "low", '
        f'"reason": "<one line>"}}'
        for c in criteria
    )
    return f"""You are scoring a research assistant's answer against a rubric.

Question: {question}

Answer:
{final_answer}

{_build_reference_block(expected)}Criteria:
{criteria_text}

For each criterion, first write your reasoning (point at the parts of the answer that justify the score), then the score, then your confidence: "low" if you cannot judge it reliably, otherwise "high".

Respond with JSON only, in exactly this shape and key order:
{{{shape}, "summary": "<one sentence>"}}
Do not include any other text."""


async def run_quality_judge(
    question: str,
    final_answer: str,
    criteria: list[dict],
    judge_model: str | None,
    expected: ExpectedOutput | None = None,
) -> QualityVerdict | None:
    """Ask an LLM judge to score final_answer against quality_criteria.

    The judge is prompted for a QualityVerdict (reasoning → score → confidence
    per criterion) and its response is validated against that model. When the
    dataset item's `expected` is given, its key points and scoring hints are
    added to the prompt as the judge's reference. The
    `ChatLiteLLM` client is used without `.with_structured_output()`: the
    default local backends (llama.cpp / Unsloth Studio via LiteLLM) don't reliably
    support tool-calling-based structured output, so the schema is enforced by
    validation instead — see PLAN.md Phase 5b.

    Returns:
        The validated verdict, or None if the judge call failed or its response
        didn't parse. Callers must treat None as "no score available" — a
        missing score, not a failing one — since a judge outage is an
        infrastructure failure, not a quality signal.
    """
    llm = build_llm(NodeLLMConfig(model=judge_model)) if judge_model else build_llm()
    prompt = _build_rubric_prompt(question, final_answer, criteria, expected)
    try:
        response = await llm_invoke_with_retry(llm, [SystemMessage(content=prompt)], _JUDGE_CONFIG)
    except LLMError as exc:
        logger.warning("quality_judge.call_failed", error=str(exc))
        return None

    raw = str(response.content)
    verdict = parse_structured(raw, QualityVerdict)
    if verdict is None:
        logger.warning("quality_judge.unparseable_response", raw_length=len(raw))
    return verdict


def make_quality_score_evaluator(criteria: list[dict], judge_model: str | None, minimum_total: int) -> Callable:
    """Return an evaluator closing over the rubric, checked once at config-parse time.

    Value is total / max_total. Pass/fail, criteria under their minimum, and
    criteria the judge flagged as low-confidence go in the comment and metadata
    — a low-confidence verdict is surfaced for human review, never dropped.
    """
    check_rubric_matches_verdict(criteria)

    async def evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
        _ = kwargs
        out = output or {}
        final_answer = out.get("final_answer")
        if not final_answer:
            return Evaluation(name="quality_score", value=0.0, comment="no final_answer produced", data_type="NUMERIC")
        expected = ExpectedOutput.model_validate(expected_output) if expected_output else None
        verdict = await run_quality_judge(out.get("question", ""), final_answer, criteria, judge_model, expected)
        if verdict is None:
            return Evaluation(name="quality_score", value=0.0, comment="quality judge call failed", data_type="NUMERIC")
        result = score_verdict(verdict, criteria, minimum_total)
        parts = [f"{'PASS' if result.passed else 'FAIL'} {result.total}/{result.max_total}"]
        if result.below_minimum:
            parts.append("below minimum: " + ", ".join(result.below_minimum))
        if result.low_confidence:
            parts.append("REVIEW (low confidence): " + ", ".join(result.low_confidence))
        parts.append(verdict.summary)
        return Evaluation(
            name="quality_score",
            value=result.total / result.max_total if result.max_total else 0.0,
            comment=" | ".join(parts),
            metadata={
                "passed": result.passed,
                "below_minimum": result.below_minimum,
                "low_confidence": result.low_confidence,
                "scores": {name: v.model_dump() for name, v in criterion_verdicts(verdict).items()},
            },
            data_type="NUMERIC",
        )

    return evaluator
