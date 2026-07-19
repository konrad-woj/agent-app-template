"""Evaluators for evals/run_experiment.py — Phase 5d (see PLAN.md).

Four evaluators, each returning a Langfuse `Evaluation` so they plug directly
into `create_score()` when Langfuse is configured (see run_experiment.py):

  quality_score_evaluator    — LLM judge scores final_answer against
                               quality_criteria (evals/configs/scoring_rubric.yaml).
                               PROVISIONAL pre-Phase-5b — see run_quality_judge().
  trace_assertion_evaluator  — deterministic checks against checkpoint history +
                               final state (evals/trace_assertions.py). No LLM call.
  turns_to_complete_evaluator — how many /v1/chat turns the run took.
  plan_approved_evaluator    — the planner interrupt fired AND the resume
                               decision matches what the dataset item asked for.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Evaluation
from logger import get_logger

from app.exceptions import LLMError
from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm, llm_invoke_with_retry, parse_structured
from evals.models import RubricJudgeVerdict
from evals.trace_assertions import evaluate_trace_assertions

logger = get_logger(__name__)

_JUDGE_CONFIG: RunnableConfig = {"configurable": {"thread_id": "quality-judge"}}

# Judge models commonly wrap JSON in a ```json ... ``` fence despite "JSON only" instructions;
# strip it before validation so a formatting quirk isn't conflated with a genuinely bad answer.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


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


def _build_rubric_prompt(question: str, final_answer: str, criteria: list[dict]) -> str:
    criteria_text = "\n\n".join(
        f"- {c['id']} ({c['name']}, max_score={c['max_score']}): {c['description'].strip()}\n"
        + "\n".join(f"    {score}: {desc}" for score, desc in c["scoring_guide"].items())
        for c in criteria
    )
    return f"""You are scoring a research assistant's answer against a rubric.

Question: {question}

Answer:
{final_answer}

Criteria:
{criteria_text}

Respond with JSON only, matching this exact shape:
{{"scores": {{"<criterion_id>": {{"score": int, "reason": "<one sentence>"}}, ...}}, "total": int, "passed": bool, "summary": "<one sentence>"}}
Do not include any other text."""


async def run_quality_judge(
    question: str,
    final_answer: str,
    criteria: list[dict],
    judge_model: str | None,
) -> RubricJudgeVerdict | None:
    """Ask an LLM judge to score final_answer against quality_criteria.

    PROVISIONAL — pre-Phase-5b: the judge is asked for freeform JSON matching
    RubricJudgeVerdict's shape via prompt instruction, not enforced structured
    output, and there is no reasoning-before-score ordering or per-criterion
    confidence flag yet. Phase 5b replaces this with CriterionVerdict /
    HolisticCriterionVerdict models via the LLM client's structured-output
    mechanism — see PLAN.md Phase 5b for the cited rationale.

    Returns:
        The validated verdict, or None if the judge call failed or its response
        didn't parse. Callers must treat None as "no score available" — a
        missing score, not a failing one — since a judge outage is an
        infrastructure failure, not a quality signal.
    """
    llm = build_llm(NodeLLMConfig(model=judge_model)) if judge_model else build_llm()
    prompt = _build_rubric_prompt(question, final_answer, criteria)
    try:
        response = await llm_invoke_with_retry(llm, [SystemMessage(content=prompt)], _JUDGE_CONFIG)
    except LLMError as exc:
        logger.warning("quality_judge.call_failed", error=str(exc))
        return None

    raw = _strip_code_fences(str(response.content))
    verdict = parse_structured(raw, RubricJudgeVerdict)
    if verdict is None:
        logger.warning("quality_judge.unparseable_response", raw_length=len(raw))
    return verdict


def make_quality_score_evaluator(criteria: list[dict], judge_model: str | None) -> Callable:
    """Return an evaluator closing over `criteria`, loaded once at config-parse time."""
    max_total = sum(c["max_score"] for c in criteria)

    async def evaluator(*, output: dict | None, expected_output: Any = None, **kwargs: Any) -> Evaluation:
        _ = expected_output, kwargs
        out = output or {}
        final_answer = out.get("final_answer")
        if not final_answer:
            return Evaluation(name="quality_score", value=0.0, comment="no final_answer produced", data_type="NUMERIC")
        verdict = await run_quality_judge(out.get("question", ""), final_answer, criteria, judge_model)
        if verdict is None:
            return Evaluation(name="quality_score", value=0.0, comment="quality judge call failed", data_type="NUMERIC")
        return Evaluation(
            name="quality_score",
            value=verdict.total / max_total if max_total else 0.0,
            comment=verdict.summary,
            data_type="NUMERIC",
        )

    return evaluator
