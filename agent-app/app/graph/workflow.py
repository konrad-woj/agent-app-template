"""Graph assembly for the research-assistant agent.

Exposes two functions:
  create_graph()   — returns the uncompiled StateGraph (LangGraph Studio).
  compile_graph()  — compiles with checkpointer + LLM + MCP tools (main.py lifespan).

Full graph:
  START → input_guard → planner → plan_review (interrupt) → resume_guard → react_researcher
        → writer → verify_subgraph → reflection_subgraph → output_guard → END

Dead-letter routing: every node that can raise is wrapped with @with_dead_letter.
The after() helper is used on each outgoing edge to detect dead_letter before
routing to the next planned node.  The terminal dead_letter node is added once.

Subgraph wrappers: verify and reflection subgraphs use disjoint state keys and
must be invoked via wrapper nodes that translate AgentState ↔ internal state.
"""

from collections.abc import Callable, Mapping
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.graph.nodes._dead_letter import after, dead_letter_node, with_dead_letter
from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm
from app.graph.nodes.input_guard import make_input_guard_node
from app.graph.nodes.output_guard import make_output_guard_node
from app.graph.nodes.planner import make_plan_review_node, make_planner_node
from app.graph.nodes.react_researcher import make_react_researcher_node_from_llm, make_tools_node
from app.graph.nodes.resume_guard import make_resume_guard_node
from app.graph.nodes.subgraphs.reflection import ReflectionState, build_reflection_subgraph
from app.graph.nodes.subgraphs.verification import VerificationState, build_verify_subgraph
from app.graph.nodes.writer import make_writer_node
from app.graph.state import AgentState
from app.guards.gliguard import GLiGuardClient


class _NullGuard:
    """No-op GLiGuardClient for use in LangGraph Studio (create_graph) where no model is loaded."""

    def check_input(self, text: str):  # type: ignore[return]
        from app.guards.gliguard import GuardResult

        return GuardResult(blocked=False)

    def check_output(self, text: str):  # type: ignore[return]
        from app.guards.gliguard import GuardResult

        return GuardResult(blocked=False)

    async def acheck_input(self, text: str, timeout: float):  # type: ignore[return]
        return self.check_input(text)

    async def acheck_output(self, text: str, timeout: float):  # type: ignore[return]
        return self.check_output(text)


def _react_condition(state: AgentState) -> Literal["tools", "writer"]:
    """Route after react_researcher: loop back to tools or proceed to writer.

    Routes to writer when model stops emitting tool_calls OR when react_steps
    reaches the AGENT_MAX_REACT_STEPS ceiling.
    """
    last = state["messages"][-1]
    ceiling_hit = state.get("react_steps", 0) >= settings.max_react_steps
    return "tools" if getattr(last, "tool_calls", None) and not ceiling_hit else "writer"


def _dead_letter_aware_route(state: AgentState, blocked_status: str, safe_route: str) -> str:
    """Shared body for the four conditions below: exception → dead_letter, `blocked_status` → END, else `safe_route`."""
    if state.get("dead_letter"):
        return "dead_letter"
    return "__end__" if state.get("status") == blocked_status else safe_route


def _input_guard_condition(state: AgentState) -> Literal["planner", "dead_letter", "__end__"]:
    """Route after input_guard: exception → dead_letter, blocked → END, safe → planner."""
    return _dead_letter_aware_route(state, "blocked", "planner")  # type: ignore[return-value]


def _planner_condition(state: AgentState) -> Literal["plan_review", "dead_letter", "__end__"]:
    """Route after planner: exception → dead_letter, blocked → END, safe → plan_review."""
    return _dead_letter_aware_route(state, "blocked", "plan_review")  # type: ignore[return-value]


def _plan_review_condition(state: AgentState) -> Literal["resume_guard", "dead_letter", "__end__"]:
    """Route after plan_review: exception → dead_letter, aborted → END, approved → resume_guard."""
    return _dead_letter_aware_route(state, "aborted", "resume_guard")  # type: ignore[return-value]


def _resume_guard_condition(state: AgentState) -> Literal["react_researcher", "dead_letter", "__end__"]:
    """Route after resume_guard: exception → dead_letter, blocked → END, safe → react_researcher."""
    return _dead_letter_aware_route(state, "blocked", "react_researcher")  # type: ignore[return-value]


def _make_run_verification(llm: BaseChatModel, fact_check_tool: BaseTool | None) -> Callable:
    """Return a wrapper node that maps AgentState → VerificationState → AgentState."""
    compiled = build_verify_subgraph(llm, fact_check_tool)

    @with_dead_letter("verify_subgraph")
    async def run_verification(state: AgentState, config: RunnableConfig) -> dict:
        verification_input: VerificationState = {
            "claims": state.get("claims", []),  # type: ignore[arg-type]
            "claim": "",
            "results": [],
        }
        result = await compiled.ainvoke(verification_input, config)
        return {"verification_results": result["results"], "status": "verifying"}

    return run_verification


def _make_run_reflection(llm: BaseChatModel) -> Callable:
    """Return a wrapper node that maps AgentState → ReflectionState → AgentState."""
    compiled = build_reflection_subgraph(llm)

    @with_dead_letter("reflection_subgraph")
    async def run_reflection(state: AgentState, config: RunnableConfig) -> dict:
        reflection_input: ReflectionState = {
            "draft": state.get("draft_answer", ""),
            "critique": "",
            "reflection_attempts": state.get("reflection_attempts", 0),
            "passed": False,
        }
        result = await compiled.ainvoke(reflection_input, config)
        return {
            "final_answer": result["draft"],
            "reflection_passed": result["passed"],
            "reflection_attempts": result["reflection_attempts"],
            "status": "done" if result["passed"] else "reflecting",
        }

    return run_reflection


def _build_graph(
    default_llm: BaseChatModel,
    mcp_tools: list[BaseTool],
    gliguard: GLiGuardClient | _NullGuard,
    fact_check_tool: BaseTool | None = None,
    node_llms: Mapping[str, BaseChatModel] | None = None,
) -> StateGraph:
    nlm = node_llms or {}
    graph: StateGraph = StateGraph(AgentState)

    # Nodes — each falls back to default_llm when not overridden
    graph.add_node("input_guard", make_input_guard_node(nlm.get("input_guard", default_llm), gliguard))  # type: ignore[arg-type]
    graph.add_node("planner", make_planner_node(nlm.get("planner", default_llm), gliguard))  # type: ignore[arg-type]
    graph.add_node("plan_review", make_plan_review_node())
    graph.add_node("resume_guard", make_resume_guard_node(gliguard))  # type: ignore[arg-type]
    graph.add_node(
        "react_researcher", make_react_researcher_node_from_llm(nlm.get("react_researcher", default_llm), mcp_tools)
    )
    graph.add_node("tools", make_tools_node(mcp_tools))
    graph.add_node("writer", make_writer_node(nlm.get("writer", default_llm)))
    graph.add_node(
        "verify_subgraph",
        _make_run_verification(nlm.get("verification", default_llm), fact_check_tool),
    )
    graph.add_node("reflection_subgraph", _make_run_reflection(nlm.get("reflection", default_llm)))
    graph.add_node("output_guard", make_output_guard_node(gliguard))  # type: ignore[arg-type]
    graph.add_node("dead_letter", dead_letter_node)

    # Edges
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", _input_guard_condition)
    graph.add_conditional_edges("planner", _planner_condition)
    graph.add_conditional_edges("plan_review", _plan_review_condition)
    graph.add_conditional_edges("resume_guard", _resume_guard_condition)
    graph.add_conditional_edges("react_researcher", _react_condition)
    graph.add_edge("tools", "react_researcher")
    graph.add_conditional_edges("writer", after("verify_subgraph"))
    graph.add_conditional_edges("verify_subgraph", after("reflection_subgraph"))
    graph.add_conditional_edges("reflection_subgraph", after("output_guard"))
    graph.add_conditional_edges("output_guard", after(END))
    graph.add_edge("dead_letter", END)

    return graph


def compile_simple_graph(checkpointer: AsyncPostgresSaver) -> CompiledStateGraph:
    """Compile a minimal 2-node stub graph (no LLM, no interrupt) for infrastructure tests.

    Used by Phase 1 checkpointing and time-travel tests to verify message accumulation
    and checkpoint persistence without needing a real LLM or MCP server.
    """
    from langchain_core.messages import AIMessage

    async def _stub_planner(state: AgentState, config: RunnableConfig) -> dict:
        last = next((m for m in reversed(state["messages"]) if not isinstance(m, AIMessage)), None)
        content = str(last.content) if last else ""
        return {
            "plan": [f"Research: {content}"],
            "status": "searching",
            "messages": [AIMessage(content=f"Plan: {content}")],
        }

    async def _stub_writer(state: AgentState, config: RunnableConfig) -> dict:
        answer = f"Draft: {' | '.join(state.get('plan', []))}"  # type: ignore[arg-type]
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "status": "done",
            "messages": [AIMessage(content=answer)],
        }

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("planner", _stub_planner)
    graph.add_node("writer", _stub_writer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "writer")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)


def create_graph() -> StateGraph:
    """Return the uncompiled StateGraph for LangGraph Studio.

    Studio injects its own checkpointer and MCP client, so this builds
    the graph with a default LLM, empty tool list, and a no-op guard.
    """
    return _build_graph(build_llm(), [], _NullGuard(), fact_check_tool=None)


def compile_graph(
    checkpointer: BaseCheckpointSaver,
    mcp_tools: list[BaseTool],
    gliguard: GLiGuardClient,
    node_llm_configs: dict[str, NodeLLMConfig] | None = None,
) -> CompiledStateGraph:
    """Compile the graph with a Postgres checkpointer for production use.

    Builds one LLM per unique NodeLLMConfig entry; nodes absent from
    node_llm_configs share the default LLM instance.

    Args:
        checkpointer: Async Postgres checkpointer from lifespan.
        mcp_tools: MCP tools loaded from the tool server (via load_mcp_tools()).
        gliguard: Loaded GLiGuardClient singleton from lifespan.
        node_llm_configs: Per-node LLM overrides keyed by node name.
            Valid keys: input_guard, planner, react_researcher, writer,
            verification, reflection.
    """
    default_llm = build_llm()
    fact_check_tool = next((t for t in mcp_tools if t.name == "fact_check"), None)
    node_llms = {name: build_llm(cfg) for name, cfg in (node_llm_configs or {}).items()}
    return _build_graph(default_llm, mcp_tools, gliguard, fact_check_tool, node_llms).compile(checkpointer=checkpointer)
