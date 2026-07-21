"""Node-level eval tier — Phase 5a (see PLAN.md).

Calls a single node factory directly against a hand-built minimal state, with no
FastAPI, Postgres, or MCP server involved. This isolates *which* node produced a
bad result instead of "something in the 10-node pipeline is wrong," and it's cheap
enough to run on every PR (see tests/evals/test_node_tasks.py, which runs every
task below with a mocked LLM and zero network calls).

Each task function has the signature `async def task(*, item, **kwargs) -> dict`,
matching the Langfuse `run_experiment` task contract (PLAN.md Phase 5d) so the same
task shape works whether it's invoked directly in a unit test or later wired into
`evals/run.py` (Phase 5e) to run against a real LLM for prompt/quality regression.

`item` is duck-typed: either a dataset-item-like object with an `.input` attribute
(Langfuse's `DatasetItem`) or a plain dict with an `"input"` key (as loaded from the
YAML files in evals/datasets/node/).

Deliberate scoping decision — GLiGuard is mocked permissively by default:
    Tasks for nodes that consult GLiGuard (planner, input_guard, resume_guard,
    output_guard) accept an optional `gliguard=` override, defaulting to a stand-in
    that never blocks and never flags PII. This tier isolates the *LLM-driven*
    decision each node makes (does the planner produce a sane plan, does the writer
    extract real claims) — not the GLiGuard model's own classification accuracy,
    which is covered by tests/guards/test_gliguard.py and, for the adversarial
    attack-taxonomy angle, the Phase 5c guardrail_redteam suite. Pass an explicit
    `gliguard=` to exercise a specific guard interaction (see the output_guard PII
    redaction test in tests/evals/test_node_tasks.py for an example).

Deliberate scoping decision — planner runs inside a minimal compiled graph:
    plan_review is the only node that calls `interrupt()` (planner generates and
    guards the plan; plan_review does nothing else before pausing for approval —
    see app/graph/nodes/planner.py for why they're split). Calling a node function
    directly (as every other task below does) raises a bare `RuntimeError` — not a
    graph suspension — because `interrupt()` reads the active Pregel run through a
    LangGraph-internal contextvar that only exists inside a real graph invocation.
    `make_planner_task` compiles a two-node planner → plan_review graph with
    `InMemorySaver` so `interrupt()` behaves exactly as it does in the full app,
    and reads the plan back from the suspended task's interrupt payload (state
    itself is not updated until the node returns, which never happens on the
    interrupted path).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.input_guard import make_input_guard_node
from app.graph.nodes.output_guard import make_output_guard_node
from app.graph.nodes.planner import make_plan_review_node, make_planner_node
from app.graph.nodes.resume_guard import make_resume_guard_node
from app.graph.nodes.subgraphs.verification import VerificationState, make_verifier_node
from app.graph.nodes.writer import make_writer_node
from app.graph.state import AgentState
from app.guards.gliguard import GLiGuardClient, GuardResult

_NODE_CONFIG: RunnableConfig = {"configurable": {"thread_id": "node-eval"}}


def _get_input(item: Any) -> dict:
    """Duck-type a dataset item into its `input` dict, Langfuse-object or plain-dict."""
    return item.input if hasattr(item, "input") else item["input"]


def _agent_state(**overrides: Any) -> AgentState:
    """Build a minimal AgentState, defaulting every field a node might read via .get()."""
    state: AgentState = {
        "messages": [],
        "plan": [],
        "plan_approved": False,
        "claims": [],
        "verification_results": [],
        "react_steps": 0,
        "draft_answer": "",
        "reflection_attempts": 0,
        "reflection_passed": False,
        "final_answer": "",
        "status": "planning",
        "guard_reason": None,
        "dead_letter": None,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _permissive_gliguard() -> GLiGuardClient:
    """A GLiGuardClient stand-in that never blocks and never flags PII — see module docstring.

    Sets both the sync (check_input/check_output) and async (acheck_input/
    acheck_output) methods: node code calls the async variants (see
    app/graph/nodes/input_guard.py etc.), but `MagicMock(spec=GLiGuardClient)`
    auto-creates the async methods as AsyncMock with an *independent* default
    return value — setting only the sync attributes silently leaves the async
    calls returning an unconfigured AsyncMock (truthy, unserializable), not the
    GuardResult set here. Confirmed by running the affected tests after the
    async guard interface was introduced (app/guards/gliguard.py); this is not
    a hypothetical.
    """
    guard = MagicMock(spec=GLiGuardClient)
    guard.check_input.return_value = GuardResult(blocked=False)
    guard.check_output.return_value = GuardResult(blocked=False, flagged_spans=[])
    guard.acheck_input = AsyncMock(return_value=GuardResult(blocked=False))
    guard.acheck_output = AsyncMock(return_value=GuardResult(blocked=False, flagged_spans=[]))
    return guard


def _extract_interrupt_value(snapshot: Any) -> dict | None:
    """Read the interrupt payload off a suspended task.

    Duplicated (intentionally — see below) from the identical helper in
    app/routers.py:_extract_interrupt_value. Kept local rather than imported so
    evals/ doesn't pull in the FastAPI router module (and its rate-limiter/auth
    dependencies) just to read a checkpoint snapshot. If the two drift, that's a
    signal this belongs in a shared, non-router module instead.
    """
    tasks = snapshot.tasks
    if not tasks:
        return None
    raw = getattr(tasks[0], "interrupts", [None])[0]
    if raw is None:
        return None
    return raw.value if hasattr(raw, "value") else None


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def make_planner_task(llm: BaseChatModel, gliguard: GLiGuardClient | None = None) -> Callable:
    """Return a task that runs planner → plan_review inside a minimal compiled graph (see module docstring)."""
    node = make_planner_node(llm, gliguard or _permissive_gliguard())
    review_node = make_plan_review_node()

    graph = StateGraph(AgentState)
    graph.add_node("planner", node)
    graph.add_node("plan_review", review_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", lambda state: END if state.get("status") == "blocked" else "plan_review")
    compiled = graph.compile(checkpointer=InMemorySaver())

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state = _agent_state(messages=[HumanMessage(content=inp["question"])])
        config: RunnableConfig = {"configurable": {"thread_id": f"planner-eval-{id(item)}"}}

        await compiled.ainvoke(state, config)
        snapshot = await compiled.aget_state(config)
        reached_interrupt = bool(snapshot.next)
        interrupt_value = _extract_interrupt_value(snapshot) if reached_interrupt else None

        return {
            "plan": interrupt_value["plan"] if interrupt_value else snapshot.values.get("plan", []),
            "reached_interrupt": reached_interrupt,
            "status": snapshot.values.get("status"),
            "guard_reason": snapshot.values.get("guard_reason"),
        }

    return task


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def make_writer_task(llm: BaseChatModel) -> Callable:
    node = make_writer_node(llm)

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state = _agent_state(
            messages=[HumanMessage(content=inp["question"])],
            plan=inp.get("plan", []),
        )
        result = await node(state, _NODE_CONFIG)
        return {
            "draft_answer": result.get("draft_answer", ""),
            "claims": result.get("claims", []),
            "status": result.get("status"),
        }

    return task


# ---------------------------------------------------------------------------
# verifier (verify_subgraph's per-claim branch — VerificationState, not AgentState)
# ---------------------------------------------------------------------------


def make_verifier_task(llm: BaseChatModel, fact_check_tool: BaseTool | None = None) -> Callable:
    node = make_verifier_node(llm, fact_check_tool)

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state: VerificationState = {"claims": [], "claim": inp["claim"], "results": []}
        result = await node(state, _NODE_CONFIG)
        results = result.get("results", [])
        verdict = results[0] if results else {}
        return {
            "supported": verdict.get("supported"),
            "confidence": verdict.get("confidence"),
            "reason": verdict.get("reason"),
        }

    return task


# ---------------------------------------------------------------------------
# output_guard — no LLM call at all (PII redaction + deterministic claim check)
# ---------------------------------------------------------------------------


def make_output_guard_task(llm: BaseChatModel | None = None, gliguard: GLiGuardClient | None = None) -> Callable:
    """`llm` is accepted (and ignored) only to keep a uniform NODE_TASK_REGISTRY[node](llm) call site."""
    _ = llm
    node = make_output_guard_node(gliguard or _permissive_gliguard())

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state = _agent_state(
            final_answer=inp.get("final_answer", ""),
            draft_answer=inp.get("final_answer", ""),
            verification_results=inp.get("verification_results", []),
        )
        result = await node(state, _NODE_CONFIG)
        return {
            "status": result.get("status"),
            "final_answer": result.get("final_answer"),
            "guard_reason": result.get("guard_reason"),
        }

    return task


# ---------------------------------------------------------------------------
# input_guard
# ---------------------------------------------------------------------------


def make_input_guard_task(llm: BaseChatModel, gliguard: GLiGuardClient | None = None) -> Callable:
    node = make_input_guard_node(llm, gliguard or _permissive_gliguard())

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state = _agent_state(messages=[HumanMessage(content=inp["message"])])
        result = await node(state, _NODE_CONFIG)
        return {"status": result.get("status"), "guard_reason": result.get("guard_reason")}

    return task


# ---------------------------------------------------------------------------
# resume_guard — no LLM call at all (regex + GLiGuard only)
# ---------------------------------------------------------------------------


def make_resume_guard_task(llm: BaseChatModel | None = None, gliguard: GLiGuardClient | None = None) -> Callable:
    """`llm` is accepted (and ignored) only to keep a uniform NODE_TASK_REGISTRY[node](llm) call site."""
    _ = llm
    node = make_resume_guard_node(gliguard or _permissive_gliguard())

    async def task(*, item: Any, **kwargs: Any) -> dict:
        _ = kwargs
        inp = _get_input(item)
        state = _agent_state(messages=[HumanMessage(content=inp["message"])])
        result = await node(state, _NODE_CONFIG)
        # resume_guard returns {} (no "status" key) on the pass-through path.
        return {"status": result.get("status", "passed"), "guard_reason": result.get("guard_reason")}

    return task


NODE_TASK_REGISTRY: dict[str, Callable[[BaseChatModel], Callable]] = {
    "planner": make_planner_task,
    "writer": make_writer_task,
    "verifier": make_verifier_task,
    "output_guard": make_output_guard_task,
    "input_guard": make_input_guard_task,
    "resume_guard": make_resume_guard_task,
}
