"""Deterministic trace-assertion engine — Phase 5d (see PLAN.md).

Evaluates the `trace_assertions` block of evals/configs/scoring_rubric.yaml
against a run's checkpoint history and final state. No LLM call — a failure
here means a wiring/routing bug, not a quality problem.

Inputs are plain dicts/lists rather than LangGraph objects so this module has
no dependency on a live graph, Postgres, or app.* — every assertion condition
is unit-testable with hand-built fixtures (see tests/evals/test_trace_assertions.py).

A `history` entry has the shape:
    {"step": int, "source": str, "next": tuple[str, ...], "values": dict}
mirroring what `run_experiment.py` builds from `graph.aget_state_history()`.
`final_state` is `history[0]["values"]` (the newest checkpoint) — AgentState
fields are cumulative/last-write-wins and are never cleared, so the final
checkpoint is a valid source for every `state_field` assertion, not just ones
that literally target the terminal node.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_NON_EMPTY = "non_empty"  # sentinel used in scoring_rubric.yaml's `expected` values


@dataclass
class TraceAssertionResult:
    id: str
    passed: bool
    detail: str


def _node_scheduled(history: list[dict], node: str) -> bool:
    """True if `node` ever appeared in a checkpoint's `next` — i.e. it was scheduled to run."""
    return any(node in snapshot.get("next", ()) for snapshot in history)


def _matches_expected(actual: Any, expected: Any) -> bool:
    if expected == _NON_EMPTY:
        return bool(actual)
    if expected is None:
        return actual is None
    return actual == expected


def _cond_equals(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    fields: list[str] = assertion["fields"]
    expected: dict = assertion["expected"]
    mismatches = []
    for field in fields:
        actual = final_state.get(field)
        exp = expected.get(field)
        if not _matches_expected(actual, exp):
            mismatches.append(f"{field}: expected {exp!r}, got {actual!r}")
    return (not mismatches, "; ".join(mismatches))


def _cond_node_executed(assertion: dict, history: list[dict], _final_state: dict) -> tuple[bool, str]:
    node = assertion["expected"]
    ok = _node_scheduled(history, node)
    return ok, "" if ok else f"'{node}' never appeared in any checkpoint's next"


def _cond_graph_interrupted_at(assertion: dict, history: list[dict], _final_state: dict) -> tuple[bool, str]:
    node = assertion["expected"]
    ok = any(snapshot.get("next") == (node,) for snapshot in history)
    return ok, "" if ok else f"no checkpoint found suspended with next == ('{node}',)"


def _cond_non_empty_list(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    field = assertion["fields"][0]
    value = final_state.get(field)
    ok = isinstance(value, list) and len(value) > 0
    return ok, "" if ok else f"{field} is not a non-empty list (got {value!r})"


def _cond_gte(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    field = assertion["fields"][0]
    value = final_state.get(field, 0)
    expected = assertion["expected"]
    ok = value >= expected
    return ok, "" if ok else f"{field}={value!r} is not >= {expected!r}"


def _cond_min_length(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    field = assertion["fields"][0]
    value = final_state.get(field, "")
    expected = assertion["expected"]
    ok = len(value or "") >= expected
    return ok, "" if ok else f"len({field})={len(value or '')} is not >= {expected}"


def _cond_one_of(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    field = assertion["fields"][0]
    value = final_state.get(field)
    expected: list = assertion["expected"]
    ok = value in expected
    return ok, "" if ok else f"{field}={value!r} not in {expected!r}"


def _cond_when_field_true_then_non_empty(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    expected = assertion["expected"]
    flag_field = expected["flag_field"]
    non_empty_field = expected["non_empty_field"]
    if not final_state.get(flag_field):
        return True, f"{flag_field} is falsy — assertion vacuously satisfied"
    ok = bool(final_state.get(non_empty_field))
    return ok, "" if ok else f"{flag_field} is true but {non_empty_field} is empty"


def _cond_all_nodes_executed(assertion: dict, history: list[dict], _final_state: dict) -> tuple[bool, str]:
    expected: list[str] = assertion["expected"]
    missing = [n for n in expected if not _node_scheduled(history, n)]
    return (not missing, f"never scheduled: {missing}" if missing else "")


def _cond_contains_message_type(assertion: dict, _history: list[dict], final_state: dict) -> tuple[bool, str]:
    expected_type = assertion["expected"]
    messages = final_state.get("messages", [])
    ok = any(getattr(m, "type", None) == expected_type for m in messages)
    return ok, "" if ok else f"no message with type={expected_type!r} found among {len(messages)} messages"


def _cond_last_ai_message_has_no_tool_calls(
    assertion: dict, _history: list[dict], final_state: dict
) -> tuple[bool, str]:
    messages = final_state.get("messages", [])
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    if last_ai is None:
        return False, "no AI message found in final state"
    tool_calls = getattr(last_ai, "tool_calls", None)
    ok = not tool_calls
    return ok, "" if ok else f"last AI message still has tool_calls: {tool_calls}"


_CONDITIONS: dict[str, Callable[[dict, list[dict], dict], tuple[bool, str]]] = {
    "equals": _cond_equals,
    "node_executed": _cond_node_executed,
    "graph_interrupted_at": _cond_graph_interrupted_at,
    "non_empty_list": _cond_non_empty_list,
    "gte": _cond_gte,
    "min_length": _cond_min_length,
    "one_of": _cond_one_of,
    "when_field_true_then_non_empty": _cond_when_field_true_then_non_empty,
    "all_nodes_executed": _cond_all_nodes_executed,
    "contains_message_type": _cond_contains_message_type,
    "last_ai_message_has_no_tool_calls": _cond_last_ai_message_has_no_tool_calls,
}


def evaluate_trace_assertions(
    assertions: list[dict],
    history: list[dict],
    final_state: dict,
    *,
    include_negative_cases: bool = False,
) -> list[TraceAssertionResult]:
    """Run every applicable assertion against one run's history + final state.

    Args:
        assertions: The `trace_assertions` list loaded from scoring_rubric.yaml.
        history: Checkpoints newest-first, each `{"step", "source", "next", "values"}`.
        final_state: `history[0]["values"]` — the run's terminal AgentState.
        include_negative_cases: Only evaluate `negative_case: true` assertions when the
            dataset item under test is itself a deliberately-adversarial negative case
            (e.g. an off-topic probe, a reject-the-plan run). Every item in
            evals/datasets/sample.yaml is a happy-path scenario, so this defaults to
            False; the Phase 5c guardrail_redteam suite is the intended home for
            adversarial cases.

    Raises:
        KeyError: If an assertion's `condition` isn't in `_CONDITIONS` — a config typo
            should fail loudly, not silently skip the assertion.
    """
    results: list[TraceAssertionResult] = []
    for assertion in assertions:
        is_negative = assertion.get("negative_case", False)
        if is_negative != include_negative_cases:
            continue
        condition = assertion["condition"]
        if condition not in _CONDITIONS:
            raise KeyError(f"Unknown trace_assertion condition {condition!r} in assertion {assertion['id']!r}")
        passed, detail = _CONDITIONS[condition](assertion, history, final_state)
        results.append(TraceAssertionResult(id=assertion["id"], passed=passed, detail=detail))
    return results
