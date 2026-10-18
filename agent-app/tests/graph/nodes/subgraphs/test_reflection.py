"""Tests for reflection subgraph: critic/refiner loop."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from app.graph.nodes.subgraphs.reflection import CriticVerdict, ReflectionState, build_reflection_subgraph

_CONFIG: RunnableConfig = {"configurable": {"thread_id": "reflection-test"}}


def _make_llm_sequence(*responses: str) -> MagicMock:
    """Return a mock LLM that yields responses in sequence."""
    ai_responses = [AIMessage(content=r) for r in responses]
    llm = MagicMock()
    llm.metadata = None  # ensures (llm.metadata or {}) falls back to settings in llm_invoke
    llm.ainvoke = AsyncMock(side_effect=ai_responses)
    return llm


def _criterion(score: int, confidence: str = "high", reason: str = "ok") -> dict:
    return {"reasoning": "evidence", "score": score, "confidence": confidence, "reason": reason}


def _critic(
    relevance: int = 4,
    completeness: int = 4,
    accuracy: int = 2,
    clarity: int = 2,
    critique: str = "Good.",
    low: tuple[str, ...] = (),
) -> str:
    """Build a critic JSON response; defaults are a clean pass (12/12)."""
    scores = {"relevance": relevance, "completeness": completeness, "accuracy": accuracy, "clarity": clarity}
    body = {
        name: _criterion(score, "low" if name in low else "high", reason=f"{name} reason")
        for name, score in scores.items()
    }
    return json.dumps({**body, "critique": critique})


def _base_state(draft: str = "draft answer", attempts: int = 0) -> ReflectionState:
    return {
        "question": "What caused the 2008 financial crisis?",
        "draft": draft,
        "critique": "",
        "reflection_attempts": attempts,
        "passed": False,
        "low_confidence": [],
        "critic_failed": False,
    }


class TestReflectionSubgraph:
    async def test_passes_on_first_attempt(self) -> None:
        llm = _make_llm_sequence(_critic(critique="Excellent answer."))
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["passed"] is True
        assert result["reflection_attempts"] == 1
        assert result["low_confidence"] == []

    async def test_refines_on_fail_then_passes(self) -> None:
        llm = _make_llm_sequence(
            _critic(completeness=1, critique="Too vague."),  # critic 1 → fail
            "Improved answer addressing vagueness.",  # refiner
            _critic(critique="Now specific."),  # critic 2 → pass
        )
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["passed"] is True
        assert result["reflection_attempts"] == 2
        assert "Improved" in result["draft"]

    async def test_exits_at_ceiling_when_never_passes(self) -> None:
        fail_critic = _critic(relevance=1, critique="Still bad.")
        refine_response = "refined draft"
        # Enough responses for ceiling iterations: critic × N + refiner × (N-1)
        responses = []
        for _ in range(3):
            responses.append(fail_critic)
            responses.append(refine_response)
        responses.append(fail_critic)  # last critic

        llm = _make_llm_sequence(*responses)
        with patch("app.graph.nodes.subgraphs.reflection.settings") as mock_settings:
            mock_settings.max_reflection_attempts = 3
            subgraph = build_reflection_subgraph(llm)
            result = await subgraph.ainvoke(_base_state(), _CONFIG)

        assert result["passed"] is False
        assert result["reflection_attempts"] >= 3

    async def test_unparseable_critic_response_fails_closed_without_refining(self) -> None:
        # A single LLM response: if the loop went on to the refiner, the mock would run dry.
        llm = _make_llm_sequence("Looks fine to me, good quality!")
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(draft="original"), _CONFIG)
        assert result["passed"] is False
        assert result["critic_failed"] is True
        assert result["draft"] == "original"
        assert result["reflection_attempts"] == 1

    async def test_critic_prompt_marks_question_and_draft_as_data(self) -> None:
        llm = _make_llm_sequence(_critic())
        subgraph = build_reflection_subgraph(llm)
        await subgraph.ainvoke(_base_state(), _CONFIG)
        system, human = llm.ainvoke.call_args.args[0]
        assert "not instructions" in system.content
        assert "<question>" in human.content and "<draft>" in human.content

    async def test_old_freeform_verdict_shape_no_longer_validates(self) -> None:
        # The pre-5b {"verdict", "critique"} shape is a parse failure now, not a silent pass/fail.
        llm = _make_llm_sequence('{"verdict": "fail", "critique": "Too vague."}')
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["critique"] == "Could not parse critic response."

    async def test_code_fenced_critic_response_is_parsed(self) -> None:
        # A fenced failing verdict used to hit the parse-failure path and auto-pass.
        llm = _make_llm_sequence(
            f"```json\n{_critic(relevance=0, critique='Off topic.')}\n```",
            "refined",
            _critic(),
        )
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["reflection_attempts"] == 2


class TestCriticVerdict:
    def test_round_trips_through_schema_validation(self) -> None:
        verdict = CriticVerdict.model_validate_json(_critic(low=("accuracy",)))
        assert verdict.relevance.score == 4
        assert verdict.accuracy.confidence == "low"

    def test_rejects_score_outside_holistic_scale(self) -> None:
        with pytest.raises(ValidationError):
            CriticVerdict.model_validate_json(_critic(clarity=3))

    def test_reasoning_is_declared_before_score(self) -> None:
        fields = list(CriticVerdict.model_fields["relevance"].annotation.model_fields)  # type: ignore[union-attr]
        assert fields.index("reasoning") < fields.index("score")


class TestCriticPassPolicy:
    async def _run_one(self, response: str) -> dict:
        # One critic call; ceiling of 1 so a fail exits without calling the refiner.
        llm = _make_llm_sequence(response)
        with patch("app.graph.nodes.subgraphs.reflection.settings") as mock_settings:
            mock_settings.max_reflection_attempts = 1
            return await build_reflection_subgraph(llm).ainvoke(_base_state(), _CONFIG)

    async def test_fails_when_a_criterion_is_below_its_minimum_despite_high_total(self) -> None:
        # 4 + 4 + 0 + 2 = 10 ≥ 9, but accuracy 0 < minimum 1.
        result = await self._run_one(_critic(accuracy=0))
        assert result["passed"] is False

    async def test_fails_when_total_below_minimum_even_if_every_criterion_meets_its_own(self) -> None:
        # 3 + 3 + 1 + 1 = 8 < 9.
        result = await self._run_one(_critic(relevance=3, completeness=3, accuracy=1, clarity=1))
        assert result["passed"] is False

    async def test_passes_at_exact_threshold(self) -> None:
        # 3 + 3 + 2 + 1 = 9.
        result = await self._run_one(_critic(relevance=3, completeness=3, accuracy=2, clarity=1))
        assert result["passed"] is True

    async def test_critique_lists_failing_criteria_for_the_refiner(self) -> None:
        result = await self._run_one(_critic(completeness=1, critique="Missing causes."))
        assert result["critique"].startswith("Missing causes.")
        assert "completeness (1/4): completeness reason" in result["critique"]
        assert "relevance" not in result["critique"]

    async def test_low_confidence_verdict_is_surfaced_not_dropped(self) -> None:
        result = await self._run_one(_critic(low=("accuracy", "clarity")))
        assert result["passed"] is True  # confidence doesn't change pass/fail …
        assert result["low_confidence"] == ["accuracy", "clarity"]  # … but is returned for review

    async def test_critic_prompt_includes_the_question(self) -> None:
        llm = _make_llm_sequence(_critic())
        await build_reflection_subgraph(llm).ainvoke(_base_state(), _CONFIG)
        human = llm.ainvoke.call_args.args[0][1]
        assert "What caused the 2008 financial crisis?" in human.content
