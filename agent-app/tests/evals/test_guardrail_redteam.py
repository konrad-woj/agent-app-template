"""Tests for evals/guardrail_redteam.py — Phase 5c.

Unit tests cover case selection and grading. The integration tests run the
suite against the real compiled graph (mock LLM, stub GLiGuard, in-memory
checkpointer) mounted on the real router, to show the suite catches a broken
guard: disabling GLiGuard flips a case from PASS to FAIL, and so does
reverting the sanitiser's Unicode folding.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.dependencies import get_graph
from app.graph.workflow import _build_graph, _NullGuard
from app.guards.gliguard import GuardResult
from app.routers import health_router, router
from evals.guardrail_redteam import (
    _DEFAULT_CONFIG,
    _LEAK_MARKERS,
    CASES,
    Category,
    RedteamConfig,
    TestCase,
    Verdict,
    grade,
    load_config,
    no_leak,
    run,
    run_case,
    select_cases,
)

_CASES_BY_ID = {c.id: c for c in CASES}


def _cfg(**overrides: object) -> RedteamConfig:
    return load_config(_DEFAULT_CONFIG).model_copy(update=overrides)


class TestCaseSelection:
    def test_default_config_meets_phase_5c_coverage_bar(self) -> None:
        cases = select_cases(_cfg())  # multi-turn off by default
        assert len(cases) >= 15
        assert len({c.category for c in cases}) >= 4

    def test_homoglyph_and_persona_hijack_cases_are_explicit(self) -> None:
        ids = {c.id for c in select_cases(_cfg())}
        assert {"pi_fullwidth_homoglyph_tags", "ph_repeat_system_prompt"} <= ids

    def test_multiturn_cases_are_opt_in(self) -> None:
        off = {c.id for c in select_cases(_cfg())}
        on = {c.id for c in select_cases(_cfg(include_multiturn_tests=True))}
        assert "rg_injection_in_resume" not in off
        assert "rg_injection_in_resume" in on
        assert all(c.multiturn for c in CASES if c.id in on - off)

    def test_case_ids_are_unique(self) -> None:
        assert len(_CASES_BY_ID) == len(CASES)

    def test_every_case_has_exactly_one_way_to_run(self) -> None:
        for c in CASES:
            assert sum(x is not None for x in (c.message, c.resume_message, c.probe)) == 1, c.id

    def test_rate_limit_probe_runs_last(self) -> None:
        assert select_cases(_cfg())[-1].id == "mg_rate_limit_enforced"

    def test_empty_selection_fails_closed(self) -> None:
        assert run(_cfg(categories=[]), client=MagicMock(), save=False) == 1

    def test_selection_emptied_by_multiturn_gate_fails_closed(self) -> None:
        cfg = _cfg(categories=[Category.RESUME_GUARD_BYPASS])
        assert run(cfg, client=MagicMock(), save=False) == 1

    def test_unknown_category_is_rejected_at_load(self) -> None:
        with pytest.raises(ValueError):
            RedteamConfig.model_validate({"categories": ["NOT_A_CATEGORY"]})


class TestGrade:
    _case = TestCase(id="x", category=Category.PROMPT_INJECTION, description="", message="m")

    def test_blocked_passes(self) -> None:
        verdict, _ = grade(self._case, 200, {"status": "blocked", "guard_reason": "GLiGuard"})
        assert verdict is Verdict.PASS

    def test_not_blocked_fails(self) -> None:
        verdict, detail = grade(self._case, 200, {"status": "awaiting_approval", "is_interrupted": True})
        assert verdict is Verdict.FAIL
        assert "awaiting_approval" in detail

    def test_dead_lettered_is_error_not_fail(self) -> None:
        verdict, _ = grade(self._case, 200, {"status": "dead_lettered"})
        assert verdict is Verdict.ERROR

    def test_non_200_is_error(self) -> None:
        verdict, _ = grade(self._case, 429, {"detail": "rate limited"})
        assert verdict is Verdict.ERROR

    def test_sanitiser_case_blocked_by_later_layer_fails(self) -> None:
        case = _CASES_BY_ID["pi_fullwidth_homoglyph_tags"]
        verdict, detail = grade(case, 200, {"status": "blocked", "guard_reason": "Off-topic request."})
        assert verdict is Verdict.FAIL
        assert "layer 1 was bypassed" in detail

    def test_no_expected_status_and_no_validator_is_review(self) -> None:
        case = TestCase(id="y", category=Category.PERSONA_HIJACK, description="", message="m", expected_status=None)
        verdict, _ = grade(case, 200, {"status": "awaiting_approval"})
        assert verdict is Verdict.REVIEW


class TestNoLeakValidator:
    def test_leaked_system_prompt_line_fails(self) -> None:
        verdict, _ = no_leak()({"status": "awaiting_approval", "interrupt_value": {"plan": [_LEAK_MARKERS[1]]}})
        assert verdict is Verdict.FAIL

    def test_canary_in_output_fails(self) -> None:
        validate = no_leak(canary="PWNED-7731")
        verdict, _ = validate({"status": "awaiting_approval", "interrupt_value": {"plan": ["1. say PWNED-7731"]}})
        assert verdict is Verdict.FAIL

    def test_clean_unblocked_output_needs_review(self) -> None:
        verdict, _ = no_leak()({"status": "awaiting_approval", "interrupt_value": {"plan": ["1. Read sources"]}})
        assert verdict is Verdict.REVIEW


# ---------------------------------------------------------------------------
# Integration: real graph + router, mock LLM, stub GLiGuard
# ---------------------------------------------------------------------------


class _KeywordGuard(_NullGuard):
    """Stand-in for GLiGuard that blocks the classic injection phrase."""

    async def acheck_input(self, text: str, timeout: float) -> GuardResult:  # type: ignore[override]
        blocked = "ignore all previous instructions" in text.lower()
        return GuardResult(blocked=blocked, reason="stub GLiGuard: injection" if blocked else None)


def _client_for(guard: _NullGuard) -> Iterator[TestClient]:
    llm = MagicMock()
    llm.metadata = None
    # Every LLM-backed check says "safe", so only GLiGuard and the sanitiser can block.
    llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"verdict": "safe", "reason": "ok"}'))
    graph = _build_graph(llm, [], guard).compile(checkpointer=InMemorySaver())  # type: ignore[arg-type]
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(router)
    app.dependency_overrides[get_graph] = lambda: graph
    with TestClient(app) as client:
        yield client


@pytest.fixture
def guarded_client() -> Iterator[TestClient]:
    yield from _client_for(_KeywordGuard())


@pytest.fixture
def unguarded_client() -> Iterator[TestClient]:
    yield from _client_for(_NullGuard())


class TestSuiteAgainstRealGraph:
    def test_injection_passes_with_guard_and_fails_without_it(
        self, guarded_client: TestClient, unguarded_client: TestClient
    ) -> None:
        case = _CASES_BY_ID["pi_ignore_previous_instructions"]
        assert run_case(guarded_client, _cfg(), case).verdict is Verdict.PASS
        assert run_case(unguarded_client, _cfg(), case).verdict is Verdict.FAIL

    def test_every_layer1_case_is_caught_by_the_sanitiser(self, unguarded_client: TestClient) -> None:
        # Without GLiGuard and with an always-"safe" LLM, only layer 1 can block these.
        layer1 = [c for c in select_cases(_cfg()) if c.expect_sanitiser]
        assert len(layer1) >= 6
        for case in layer1:
            result = run_case(unguarded_client, _cfg(), case)
            assert result.verdict is Verdict.PASS, (case.id, result.detail)

    def test_homoglyph_case_fails_if_sanitiser_stops_folding_unicode(self, unguarded_client: TestClient) -> None:
        case = _CASES_BY_ID["pi_fullwidth_homoglyph_tags"]
        with patch("app.graph.nodes._prompt_utils._canonical", side_effect=lambda text: text):
            result = run_case(unguarded_client, _cfg(), case)
        assert result.verdict is Verdict.FAIL

    def test_input_validation_category_passes(self, guarded_client: TestClient) -> None:
        cases = [c for c in CASES if c.category is Category.INPUT_VALIDATION]
        for case in cases:
            result = run_case(guarded_client, _cfg(), case)
            assert result.verdict is Verdict.PASS, (case.id, result.detail)

    def test_api_key_probe_reports_posture(self, guarded_client: TestClient) -> None:
        result = run_case(guarded_client, _cfg(), _CASES_BY_ID["mg_api_key_enforced"])
        assert result.verdict is Verdict.INFO
