"""Tests for evals/evaluators.py — Phase 5d.

quality_score_evaluator's judge LLM call is mocked (no network) — see
_mock_llm below. trace_assertion_evaluator, turns_to_complete_evaluator, and
plan_approved_evaluator are pure functions over a plain `output` dict and
need no mocking at all.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from evals.evaluators import (
    make_quality_score_evaluator,
    make_trace_assertion_evaluator,
    plan_approved_evaluator,
    run_quality_judge,
    turns_to_complete_evaluator,
)

_CRITERIA = [
    {
        "id": "specificity",
        "name": "Specificity",
        "max_score": 3,
        "description": "How specific the answer is.",
        "scoring_guide": {0: "vague", 3: "specific"},
    },
]


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.metadata = None
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


class TestTurnsToCompleteEvaluator:
    def test_reports_turns_taken(self) -> None:
        ev = turns_to_complete_evaluator(output={"turns": 2})
        assert ev.value == 2.0
        assert ev.data_type == "NUMERIC"

    def test_zero_when_turns_missing(self) -> None:
        ev = turns_to_complete_evaluator(output={})
        assert ev.value == 0.0


class TestPlanApprovedEvaluator:
    def test_fails_when_interrupt_never_reached(self) -> None:
        ev = plan_approved_evaluator(output={"reached_interrupt": False, "status": "blocked"})
        assert ev.value == 0

    def test_passes_when_approved_and_not_aborted(self) -> None:
        ev = plan_approved_evaluator(output={"reached_interrupt": True, "approve_plan": True, "status": "done"})
        assert ev.value == 1

    def test_fails_when_approved_but_run_was_aborted(self) -> None:
        ev = plan_approved_evaluator(output={"reached_interrupt": True, "approve_plan": True, "status": "aborted"})
        assert ev.value == 0

    def test_passes_when_rejected_and_status_is_aborted(self) -> None:
        ev = plan_approved_evaluator(output={"reached_interrupt": True, "approve_plan": False, "status": "aborted"})
        assert ev.value == 1

    def test_fails_when_rejected_but_status_is_not_aborted(self) -> None:
        ev = plan_approved_evaluator(output={"reached_interrupt": True, "approve_plan": False, "status": "done"})
        assert ev.value == 0


class TestTraceAssertionEvaluator:
    def test_reports_pass_rate_and_lists_failures(self) -> None:
        assertions = [
            {"id": "a1", "condition": "equals", "fields": ["status"], "expected": {"status": "done"}},
            {"id": "a2", "condition": "non_empty_list", "fields": ["plan"]},
        ]
        evaluator = make_trace_assertion_evaluator(assertions)
        output = {
            "history": [{"step": 0, "source": "loop", "next": (), "values": {"status": "done", "plan": []}}],
            "final_state": {"status": "done", "plan": []},
        }
        ev = evaluator(output=output)
        assert ev.value == 0.5
        assert "a2" in ev.comment

    def test_zero_when_no_history_available(self) -> None:
        evaluator = make_trace_assertion_evaluator([{"id": "a1", "condition": "gte", "fields": ["x"], "expected": 1}])
        ev = evaluator(output={"history": [], "final_state": {}})
        assert ev.value == 0.0
        assert "history" in ev.comment


class TestRunQualityJudge:
    async def test_returns_none_on_judge_parse_failure(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm("not json")):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is None

    async def test_returns_parsed_verdict_on_valid_json(self) -> None:
        payload = '{"scores": {"specificity": {"score": 3, "reason": "very specific"}}, "total": 3, "passed": true, "summary": "great"}'
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(payload)):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is not None
        assert verdict.total == 3
        assert verdict.passed is True

    async def test_returns_parsed_verdict_when_wrapped_in_markdown_code_fence(self) -> None:
        payload = (
            '```json\n{"scores": {"specificity": {"score": 3, "reason": "very specific"}}, '
            '"total": 3, "passed": true, "summary": "great"}\n```'
        )
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(payload)):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is not None
        assert verdict.total == 3
        assert verdict.passed is True


class TestQualityScoreEvaluator:
    async def test_zero_when_no_final_answer(self) -> None:
        evaluator = make_quality_score_evaluator(_CRITERIA, judge_model=None)
        ev = await evaluator(output={"final_answer": None})
        assert ev.value == 0.0

    async def test_scores_fraction_of_max_total(self) -> None:
        payload = '{"scores": {"specificity": {"score": 3, "reason": "very specific"}}, "total": 3, "passed": true, "summary": "great"}'
        evaluator = make_quality_score_evaluator(_CRITERIA, judge_model=None)
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(payload)):
            ev = await evaluator(output={"final_answer": "some answer", "question": "q"})
        assert ev.value == 1.0

    async def test_zero_when_judge_fails(self) -> None:
        evaluator = make_quality_score_evaluator(_CRITERIA, judge_model=None)
        with patch("evals.evaluators.build_llm", return_value=_mock_llm("not json")):
            ev = await evaluator(output={"final_answer": "some answer", "question": "q"})
        assert ev.value == 0.0
        assert "judge" in ev.comment.lower()
