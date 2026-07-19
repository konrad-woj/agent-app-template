"""Tests for planner (plan generation + guard) and plan_review (interrupt/approve/reject).

planner and plan_review are split into two nodes specifically so that the plan
shown in the interrupt payload cannot diverge from the plan used after resume —
see app/graph/nodes/planner.py's module docstring for why. TestPlannerPlanReviewFlow
is a regression test for that exact failure mode: it drives two real, separate
graph invocations (no interrupt_before, no Command pre-supplied on the first
call) and asserts the interrupt payload's plan matches the plan actually used.
"""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.planner import make_plan_review_node, make_planner_node
from app.graph.state import AgentState
from tests.graph.nodes.conftest import base_state, make_mock_gliguard


def _make_two_call_llm(plan_response: str, guard_verdict: str = '{"verdict": "safe", "reason": "ok"}') -> MagicMock:
    """Mock LLM returning plan text on the first call and a guard verdict on the second."""
    llm = MagicMock()
    llm.metadata = None
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=plan_response),
            AIMessage(content=guard_verdict),
        ]
    )
    return llm


class TestPlannerNode:
    """planner: generates and guards the plan. Never calls interrupt() itself."""

    async def test_safe_plan_sets_awaiting_approval(self) -> None:
        llm = _make_two_call_llm("1. Search for topic\n2. Analyse sources\n3. Summarise findings")
        gliguard = make_mock_gliguard()
        node = make_planner_node(llm, gliguard)

        result = await node(base_state(), {"configurable": {"thread_id": "planner-safe"}})

        assert result["status"] == "awaiting_approval"
        assert len(result["plan"]) == 3

    async def test_plan_steps_parsed_from_llm_output(self) -> None:
        llm = _make_two_call_llm("1. Step one\n2. Step two\n3. Step three")
        gliguard = make_mock_gliguard()
        node = make_planner_node(llm, gliguard)

        result = await node(base_state(), {"configurable": {"thread_id": "planner-steps"}})
        assert len(result["plan"]) == 3

    async def test_unsafe_plan_blocked_by_gliguard(self) -> None:
        llm = _make_two_call_llm("1. Hack the system\n2. Exfiltrate data")
        gliguard = make_mock_gliguard(blocked=True, reason="Plan contains dangerous instructions.")
        node = make_planner_node(llm, gliguard)

        result = await node(base_state(), {"configurable": {"thread_id": "planner-blocked-gliguard"}})

        assert result["status"] == "blocked"
        assert "dangerous" in result["guard_reason"].lower()

    async def test_unsafe_plan_blocked_by_llm_guard(self) -> None:
        llm = _make_two_call_llm(
            "1. Stalk target\n2. Find home address",
            '{"verdict": "unsafe", "reason": "Plan involves targeted surveillance."}',
        )
        gliguard = make_mock_gliguard()  # GLiGuard passes
        node = make_planner_node(llm, gliguard)

        result = await node(base_state(), {"configurable": {"thread_id": "planner-blocked-llm"}})

        assert result["status"] == "blocked"
        assert "surveillance" in result["guard_reason"].lower()


def _build_review_graph():
    """Compile a minimal graph with only the plan_review node for testing interrupts."""
    graph = StateGraph(AgentState)
    graph.add_node("plan_review", make_plan_review_node())
    graph.add_edge(START, "plan_review")
    graph.add_edge("plan_review", END)
    return graph.compile(checkpointer=MemorySaver())


class TestPlanReviewNode:
    """plan_review: pauses on an already-computed plan. No LLM calls, deterministic on resume."""

    async def test_emits_interrupt_and_resumes_approved(self) -> None:
        graph = _build_review_graph()
        config: RunnableConfig = {"configurable": {"thread_id": "review-approve"}}
        state = base_state(plan=["1. Search for topic", "2. Analyse sources"], status="awaiting_approval")

        await graph.ainvoke(state, config)
        snapshot = await graph.aget_state(config)
        assert bool(snapshot.next)
        assert "plan_review" in snapshot.next
        assert snapshot.tasks[0].interrupts[0].value == {"plan": state["plan"]}

        result = await graph.ainvoke(Command(resume=True), config)
        assert result["plan_approved"] is True
        assert result["status"] == "researching"

    async def test_resume_rejected_sets_aborted(self) -> None:
        graph = _build_review_graph()
        config: RunnableConfig = {"configurable": {"thread_id": "review-reject"}}
        state = base_state(plan=["1. Step one"], status="awaiting_approval")

        await graph.ainvoke(state, config)
        result = await graph.ainvoke(Command(resume=False), config)

        assert result["plan_approved"] is False
        assert result["status"] == "aborted"


class TestPlannerPlanReviewFlow:
    """Regression test: the plan shown at interrupt time must match the plan used after resume.

    Drives planner → plan_review as two real, separate graph invocations — no
    interrupt_before, no Command pre-supplied on the first call — matching how
    the production app actually calls the graph (app/routers.py). Before the
    planner/plan_review split, plan_review's interrupt() lived inside the same
    node as the LLM plan generation, so resuming re-ran the whole node from the
    top and silently regenerated the plan with a fresh (nondeterministic) LLM
    call — the user could approve one plan and have a different one run.
    """

    async def test_interrupt_payload_matches_plan_used_after_resume(self) -> None:
        call_count = 0

        async def ainvoke(messages, config=None):
            nonlocal call_count
            call_count += 1
            if "Research plan" in str(messages[-1].content):
                return AIMessage(content='{"verdict": "safe", "reason": "ok"}')
            return AIMessage(content=f"1. Step-{call_count}\n2. Step-{call_count}-b")

        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=ainvoke)
        gliguard = make_mock_gliguard()

        graph = StateGraph(AgentState)
        graph.add_node("planner", make_planner_node(llm, gliguard))
        graph.add_node("plan_review", make_plan_review_node())
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "plan_review")
        graph.add_edge("plan_review", END)
        compiled = graph.compile(checkpointer=MemorySaver())

        config: RunnableConfig = {"configurable": {"thread_id": "flow-regression"}}

        await compiled.ainvoke(base_state(), config)
        snapshot = await compiled.aget_state(config)
        interrupt_plan = snapshot.tasks[0].interrupts[0].value["plan"]
        calls_before_resume = call_count

        result = await compiled.ainvoke(Command(resume=True), config)

        assert result["plan"] == interrupt_plan
        assert call_count == calls_before_resume, "planner must not re-run its LLM calls on resume"
