"""Tests for evals/evaluators.py — Phases 5d, 5b and 7.

quality_score_evaluator's judge LLM call is mocked (no network) — see
_mock_llm below. The other evaluators are pure functions over a plain
`output` dict and need no mocking at all.
"""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from evals.evaluators import (
    check_rubric_matches_verdict,
    expected_terms_evaluator,
    make_quality_score_evaluator,
    make_trace_assertion_evaluator,
    plan_approved_evaluator,
    run_quality_judge,
    score_verdict,
    turns_to_complete_evaluator,
)
from evals.models import ExpectedOutput, QualityVerdict

# The real rubric — these tests double as a guard that the shipped config and
# QualityVerdict still agree.
_RUBRIC = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "evals" / "configs" / "scoring_rubric.yaml").read_text()
)
_CRITERIA: list[dict] = _RUBRIC["quality_criteria"]
_MIN_TOTAL: int = _RUBRIC["pass_policy"]["minimum_total"]
_FULL_MARKS = {
    "key_point_coverage": 4,
    "factual_accuracy": 4,
    "specificity": 4,
    "focus": 2,
    "uncertainty_handling": 2,
}
_EXPECTED = ExpectedOutput.model_validate(
    {
        "must_cover": [
            {"id": "kp1", "description": "Subprime mortgage lending fuelled a housing bubble"},
            {"id": "kp2", "description": "Lehman Brothers collapsed in September 2008"},
        ],
        "scoring_hints": ["Should name securitisation (MBS/CDOs) as the transmission channel"],
    }
)


def _verdict_json(low: tuple[str, ...] = (), **scores: int) -> str:
    merged = {**_FULL_MARKS, **scores}
    body = {
        cid: {"reasoning": "evidence", "score": s, "confidence": "low" if cid in low else "high", "reason": f"{cid} ok"}
        for cid, s in merged.items()
    }
    return json.dumps({**body, "summary": "great"})


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


_TERM_CHECKS = {"must_include_terms": ["subprime", "Lehman"], "must_not_contain": ["I cannot help with"]}


class TestExpectedTermsEvaluator:
    def test_all_checks_pass(self) -> None:
        answer = "Subprime lending and the Lehman collapse triggered the crisis."
        ev = expected_terms_evaluator(output={"final_answer": answer}, expected_output=_TERM_CHECKS)
        assert ev.value == 1.0

    def test_missing_required_term_is_named(self) -> None:
        ev = expected_terms_evaluator(
            output={"final_answer": "Subprime lending caused it."}, expected_output=_TERM_CHECKS
        )
        assert ev.value == 2 / 3
        assert "Lehman" in (ev.comment or "")

    def test_forbidden_phrase_found_is_named(self) -> None:
        answer = "Subprime and Lehman. I cannot help with more detail."
        ev = expected_terms_evaluator(output={"final_answer": answer}, expected_output=_TERM_CHECKS)
        assert ev.value == 2 / 3
        assert "I cannot help with" in (ev.comment or "")

    def test_matching_is_case_insensitive(self) -> None:
        answer = "SUBPRIME loans; lehman brothers. i CANNOT help WITH that."
        ev = expected_terms_evaluator(output={"final_answer": answer}, expected_output=_TERM_CHECKS)
        assert ev.value == 2 / 3  # both required terms found, forbidden phrase also found

    def test_no_term_checks_scores_one(self) -> None:
        ev = expected_terms_evaluator(output={"final_answer": "anything"}, expected_output={})
        assert ev.value == 1.0
        assert "no term checks" in (ev.comment or "")

    def test_zero_when_no_final_answer(self) -> None:
        ev = expected_terms_evaluator(output={"final_answer": None}, expected_output=_TERM_CHECKS)
        assert ev.value == 0.0


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


class TestRubricConfig:
    def test_shipped_rubric_matches_quality_verdict(self) -> None:
        check_rubric_matches_verdict(_CRITERIA)

    def test_max_total_is_16_and_minimum_total_fits(self) -> None:
        assert sum(c["max_score"] for c in _CRITERIA) == 16
        assert sum(c["per_criterion_minimum"] for c in _CRITERIA) <= _MIN_TOTAL <= 16

    def test_mismatched_max_score_raises(self) -> None:
        bad = [{**c, "max_score": 3} if c["id"] == "specificity" else c for c in _CRITERIA]
        with pytest.raises(ValueError, match="specificity"):
            check_rubric_matches_verdict(bad)

    def test_missing_and_unknown_criteria_raise(self) -> None:
        bad = [c for c in _CRITERIA if c["id"] != "focus"] + [{"id": "tone", "max_score": 2}]
        with pytest.raises(ValueError, match="focus") as exc:
            check_rubric_matches_verdict(bad)
        assert "tone" in str(exc.value)

    def test_evaluator_factory_rejects_mismatched_rubric(self) -> None:
        with pytest.raises(ValueError):
            make_quality_score_evaluator(_CRITERIA[:1], judge_model=None, minimum_total=_MIN_TOTAL)


class TestScoreVerdict:
    def test_full_marks_pass(self) -> None:
        result = score_verdict(QualityVerdict.model_validate_json(_verdict_json()), _CRITERIA, _MIN_TOTAL)
        assert (result.total, result.max_total, result.passed) == (16, 16, True)

    def test_below_per_criterion_minimum_fails_even_with_high_total(self) -> None:
        # 16 - 2 = 14 ≥ 12, but key_point_coverage 2 < minimum 3.
        verdict = QualityVerdict.model_validate_json(_verdict_json(key_point_coverage=2))
        result = score_verdict(verdict, _CRITERIA, _MIN_TOTAL)
        assert result.passed is False
        assert result.below_minimum == ["key_point_coverage"]

    def test_below_minimum_total_fails(self) -> None:
        # 3+3+2+1+1 = 10 < 12 while every criterion meets its own minimum.
        verdict = QualityVerdict.model_validate_json(
            _verdict_json(key_point_coverage=3, factual_accuracy=3, specificity=2, focus=1, uncertainty_handling=1)
        )
        result = score_verdict(verdict, _CRITERIA, _MIN_TOTAL)
        assert (result.total, result.passed, result.below_minimum) == (10, False, [])

    def test_low_confidence_criteria_are_collected(self) -> None:
        verdict = QualityVerdict.model_validate_json(_verdict_json(low=("focus",)))
        assert score_verdict(verdict, _CRITERIA, _MIN_TOTAL).low_confidence == ["focus"]

    def test_score_outside_scale_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QualityVerdict.model_validate_json(_verdict_json(focus=4))


class TestRunQualityJudge:
    async def test_returns_none_on_judge_parse_failure(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm("not json")):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is None

    async def test_returns_none_for_pre_5b_freeform_shape(self) -> None:
        payload = '{"scores": {"specificity": {"score": 3, "reason": "x"}}, "total": 3, "passed": true, "summary": "s"}'
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(payload)):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is None

    async def test_returns_parsed_verdict_on_valid_json(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(_verdict_json(specificity=3))):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is not None
        assert verdict.specificity.score == 3

    async def test_returns_parsed_verdict_when_wrapped_in_markdown_code_fence(self) -> None:
        payload = f"```json\n{_verdict_json()}\n```"
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(payload)):
            verdict = await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        assert verdict is not None

    async def test_prompt_asks_for_reasoning_before_score_for_every_criterion(self) -> None:
        llm = _mock_llm(_verdict_json())
        with patch("evals.evaluators.build_llm", return_value=llm):
            await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        prompt = llm.ainvoke.call_args.args[0][0].content
        for c in _CRITERIA:
            assert f'"{c["id"]}": {{"reasoning"' in prompt
            assert f"score 0-{c['max_score']}" in prompt

    async def test_judge_prompt_includes_key_points_and_hints(self) -> None:
        llm = _mock_llm(_verdict_json())
        with patch("evals.evaluators.build_llm", return_value=llm):
            await run_quality_judge("q", "answer", _CRITERIA, judge_model=None, expected=_EXPECTED)
        prompt = llm.ainvoke.call_args.args[0][0].content
        for point in _EXPECTED.must_cover:
            assert point.description in prompt
        for hint in _EXPECTED.scoring_hints:
            assert hint in prompt

    async def test_judge_prompt_omits_reference_block_when_no_expected_output(self) -> None:
        llm = _mock_llm(_verdict_json())
        with patch("evals.evaluators.build_llm", return_value=llm):
            await run_quality_judge("q", "answer", _CRITERIA, judge_model=None)
        prompt = llm.ainvoke.call_args.args[0][0].content
        assert "Key points" not in prompt
        assert "Scoring hints" not in prompt


class TestQualityScoreEvaluator:
    def _evaluator(self) -> Callable:
        return make_quality_score_evaluator(_CRITERIA, judge_model=None, minimum_total=_MIN_TOTAL)

    async def test_zero_when_no_final_answer(self) -> None:
        ev = await self._evaluator()(output={"final_answer": None})
        assert ev.value == 0.0

    async def test_scores_fraction_of_max_total(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(_verdict_json(specificity=0))):
            ev = await self._evaluator()(output={"final_answer": "some answer", "question": "q"})
        assert ev.value == 12 / 16
        assert ev.comment.startswith("FAIL 12/16")
        assert ev.metadata["below_minimum"] == ["specificity"]

    async def test_low_confidence_is_surfaced_in_comment_and_metadata(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm(_verdict_json(low=("uncertainty_handling",)))):
            ev = await self._evaluator()(output={"final_answer": "some answer", "question": "q"})
        assert ev.comment.startswith("PASS 16/16")
        assert "REVIEW (low confidence): uncertainty_handling" in ev.comment
        assert ev.metadata["low_confidence"] == ["uncertainty_handling"]

    async def test_passes_item_expected_output_to_the_judge(self) -> None:
        llm = _mock_llm(_verdict_json())
        with patch("evals.evaluators.build_llm", return_value=llm):
            await self._evaluator()(
                output={"final_answer": "some answer", "question": "q"},
                expected_output=_EXPECTED.model_dump(),
            )
        prompt = llm.ainvoke.call_args.args[0][0].content
        assert _EXPECTED.must_cover[0].description in prompt

    async def test_zero_when_judge_fails(self) -> None:
        with patch("evals.evaluators.build_llm", return_value=_mock_llm("not json")):
            ev = await self._evaluator()(output={"final_answer": "some answer", "question": "q"})
        assert ev.value == 0.0
        assert "judge" in ev.comment.lower()
