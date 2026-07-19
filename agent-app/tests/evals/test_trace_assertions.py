"""Tests for evals/trace_assertions.py — Phase 5d deterministic assertion engine.

Every test uses hand-built plain-dict fixtures — no Postgres, no live server,
no LangGraph graph object. Each condition is exercised on a fixture that
passes and one that's deliberately broken.
"""

from langchain_core.messages import AIMessage, ToolMessage

from evals.trace_assertions import evaluate_trace_assertions

_A_PASSES_ON_TOPIC = {
    "id": "a_pass",
    "check": "checkpoint",
    "condition": "node_executed",
    "expected": "planner",
}


def _history(*next_tuples: tuple[str, ...]) -> list[dict]:
    return [{"step": i, "source": "loop", "next": nt, "values": {}} for i, nt in enumerate(next_tuples)]


class TestEquals:
    def test_passes_when_fields_match(self) -> None:
        assertion = {
            "id": "a",
            "condition": "equals",
            "fields": ["status", "guard_reason"],
            "expected": {"status": "blocked", "guard_reason": "non_empty"},
        }
        results = evaluate_trace_assertions([assertion], [], {"status": "blocked", "guard_reason": "off-topic"})
        assert results[0].passed is True

    def test_fails_when_a_field_mismatches(self) -> None:
        assertion = {
            "id": "a",
            "condition": "equals",
            "fields": ["status"],
            "expected": {"status": "blocked"},
        }
        results = evaluate_trace_assertions([assertion], [], {"status": "done"})
        assert results[0].passed is False
        assert "status" in results[0].detail

    def test_null_sentinel_matches_none(self) -> None:
        assertion = {
            "id": "a",
            "condition": "equals",
            "fields": ["guard_reason"],
            "expected": {"guard_reason": None},
        }
        assert evaluate_trace_assertions([assertion], [], {"guard_reason": None})[0].passed is True
        assert evaluate_trace_assertions([assertion], [], {"guard_reason": "oops"})[0].passed is False


class TestNodeExecuted:
    def test_passes_when_node_appears_in_next(self) -> None:
        history = _history(("planner",), ("resume_guard",), ())
        results = evaluate_trace_assertions([_A_PASSES_ON_TOPIC], history, {})
        assert results[0].passed is True

    def test_fails_when_node_never_scheduled(self) -> None:
        history = _history(("resume_guard",), ())
        results = evaluate_trace_assertions([_A_PASSES_ON_TOPIC], history, {})
        assert results[0].passed is False


class TestGraphInterruptedAt:
    def test_passes_when_snapshot_next_is_exactly_the_node(self) -> None:
        assertion = {"id": "a", "condition": "graph_interrupted_at", "expected": "planner"}
        history = _history(("planner",))
        assert evaluate_trace_assertions([assertion], history, {})[0].passed is True

    def test_fails_when_no_snapshot_matches_exactly(self) -> None:
        assertion = {"id": "a", "condition": "graph_interrupted_at", "expected": "planner"}
        history = _history(("planner", "extra_node"))
        assert evaluate_trace_assertions([assertion], history, {})[0].passed is False


class TestNonEmptyList:
    def test_passes_on_non_empty_list(self) -> None:
        assertion = {"id": "a", "condition": "non_empty_list", "fields": ["plan"]}
        assert evaluate_trace_assertions([assertion], [], {"plan": ["step 1"]})[0].passed is True

    def test_fails_on_empty_list(self) -> None:
        assertion = {"id": "a", "condition": "non_empty_list", "fields": ["plan"]}
        assert evaluate_trace_assertions([assertion], [], {"plan": []})[0].passed is False


class TestGte:
    def test_passes_when_at_or_above_threshold(self) -> None:
        assertion = {"id": "a", "condition": "gte", "fields": ["react_steps"], "expected": 1}
        assert evaluate_trace_assertions([assertion], [], {"react_steps": 3})[0].passed is True

    def test_fails_when_below_threshold(self) -> None:
        assertion = {"id": "a", "condition": "gte", "fields": ["react_steps"], "expected": 1}
        assert evaluate_trace_assertions([assertion], [], {"react_steps": 0})[0].passed is False


class TestMinLength:
    def test_passes_when_long_enough(self) -> None:
        assertion = {"id": "a", "condition": "min_length", "fields": ["draft_answer"], "expected": 5}
        assert evaluate_trace_assertions([assertion], [], {"draft_answer": "a long draft"})[0].passed is True

    def test_fails_when_too_short(self) -> None:
        assertion = {"id": "a", "condition": "min_length", "fields": ["draft_answer"], "expected": 100}
        assert evaluate_trace_assertions([assertion], [], {"draft_answer": "short"})[0].passed is False


class TestOneOf:
    def test_passes_when_value_in_expected(self) -> None:
        assertion = {"id": "a", "condition": "one_of", "fields": ["status"], "expected": ["writing", "reflecting"]}
        assert evaluate_trace_assertions([assertion], [], {"status": "writing"})[0].passed is True

    def test_fails_when_value_not_in_expected(self) -> None:
        assertion = {"id": "a", "condition": "one_of", "fields": ["status"], "expected": ["writing", "reflecting"]}
        assert evaluate_trace_assertions([assertion], [], {"status": "blocked"})[0].passed is False


class TestWhenFieldTrueThenNonEmpty:
    def test_vacuously_true_when_flag_is_false(self) -> None:
        assertion = {
            "id": "a",
            "condition": "when_field_true_then_non_empty",
            "expected": {"flag_field": "reflection_passed", "non_empty_field": "final_answer"},
        }
        state = {"reflection_passed": False, "final_answer": ""}
        assert evaluate_trace_assertions([assertion], [], state)[0].passed is True

    def test_fails_when_flag_true_but_field_empty(self) -> None:
        assertion = {
            "id": "a",
            "condition": "when_field_true_then_non_empty",
            "expected": {"flag_field": "reflection_passed", "non_empty_field": "final_answer"},
        }
        state = {"reflection_passed": True, "final_answer": ""}
        assert evaluate_trace_assertions([assertion], [], state)[0].passed is False

    def test_passes_when_flag_true_and_field_non_empty(self) -> None:
        assertion = {
            "id": "a",
            "condition": "when_field_true_then_non_empty",
            "expected": {"flag_field": "reflection_passed", "non_empty_field": "final_answer"},
        }
        state = {"reflection_passed": True, "final_answer": "the answer"}
        assert evaluate_trace_assertions([assertion], [], state)[0].passed is True


class TestAllNodesExecuted:
    def test_passes_when_every_node_scheduled(self) -> None:
        assertion = {"id": "a", "condition": "all_nodes_executed", "expected": ["input_guard", "planner"]}
        history = _history(("input_guard",), ("planner",), ())
        assert evaluate_trace_assertions([assertion], history, {})[0].passed is True

    def test_fails_when_one_node_missing(self) -> None:
        assertion = {"id": "a", "condition": "all_nodes_executed", "expected": ["input_guard", "output_guard"]}
        history = _history(("input_guard",), ())
        result = evaluate_trace_assertions([assertion], history, {})[0]
        assert result.passed is False
        assert "output_guard" in result.detail


class TestContainsMessageType:
    def test_passes_when_a_tool_message_is_present(self) -> None:
        assertion = {"id": "a", "condition": "contains_message_type", "expected": "tool"}
        messages = [AIMessage(content="thinking"), ToolMessage(content="result", tool_call_id="1")]
        assert evaluate_trace_assertions([assertion], [], {"messages": messages})[0].passed is True

    def test_fails_when_no_tool_message_present(self) -> None:
        assertion = {"id": "a", "condition": "contains_message_type", "expected": "tool"}
        messages = [AIMessage(content="thinking")]
        assert evaluate_trace_assertions([assertion], [], {"messages": messages})[0].passed is False


class TestLastAiMessageHasNoToolCalls:
    def test_passes_when_last_ai_message_has_no_tool_calls(self) -> None:
        assertion = {"id": "a", "condition": "last_ai_message_has_no_tool_calls"}
        messages = [
            AIMessage(content="calling a tool", tool_calls=[{"name": "x", "args": {}, "id": "1"}]),
            AIMessage(content="final answer"),
        ]
        assert evaluate_trace_assertions([assertion], [], {"messages": messages})[0].passed is True

    def test_fails_when_last_ai_message_still_has_tool_calls(self) -> None:
        assertion = {"id": "a", "condition": "last_ai_message_has_no_tool_calls"}
        messages = [AIMessage(content="calling a tool", tool_calls=[{"name": "x", "args": {}, "id": "1"}])]
        assert evaluate_trace_assertions([assertion], [], {"messages": messages})[0].passed is False


class TestNegativeCaseFiltering:
    def test_negative_case_assertions_are_skipped_by_default(self) -> None:
        assertion = {
            "id": "a",
            "condition": "equals",
            "fields": ["status"],
            "expected": {"status": "aborted"},
            "negative_case": True,
        }
        results = evaluate_trace_assertions([assertion], [], {"status": "done"})
        assert results == []

    def test_negative_case_assertions_run_when_requested(self) -> None:
        assertion = {
            "id": "a",
            "condition": "equals",
            "fields": ["status"],
            "expected": {"status": "aborted"},
            "negative_case": True,
        }
        results = evaluate_trace_assertions([assertion], [], {"status": "aborted"}, include_negative_cases=True)
        assert len(results) == 1
        assert results[0].passed is True

    def test_positive_assertions_skipped_when_scanning_for_negative_cases(self) -> None:
        assertion = {"id": "a", "condition": "equals", "fields": ["status"], "expected": {"status": "done"}}
        results = evaluate_trace_assertions([assertion], [], {"status": "done"}, include_negative_cases=True)
        assert results == []


class TestUnknownCondition:
    def test_raises_on_unknown_condition(self) -> None:
        import pytest

        assertion = {"id": "a", "condition": "not_a_real_condition"}
        with pytest.raises(KeyError):
            evaluate_trace_assertions([assertion], [], {})
