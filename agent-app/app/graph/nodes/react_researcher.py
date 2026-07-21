"""ReAct researcher node: model ↔ ToolNode loop using MCP tools.

The model is called with a list of available MCP tools bound via bind_tools().
If the model emits tool_calls, the graph routes to ToolNode which executes them
and appends ToolMessages to state. The loop continues until the model stops
calling tools OR react_steps reaches the MAX_REACT_STEPS ceiling (set via
AGENT_MAX_REACT_STEPS).

react_condition() in workflow.py implements the routing — this node only
does the single model invocation.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from logger import get_logger

from app.config import settings
from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry

if TYPE_CHECKING:
    from app.graph.state import AgentState

logger = get_logger(__name__)


async def _timeout_bounded_tool_call(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """ToolNode awrap_tool_call hook: bound every MCP tool call by a timeout.

    Without this, a hung web_search/fetch_url/fact_check call (network stall,
    upstream not responding) blocks the ReAct loop indefinitely — unlike every
    LLM call in this codebase, which is already timeout-bounded. A TimeoutError
    raised here is caught by ToolNode's own error handling (same path as any
    other tool exception) and surfaced to the model as an error ToolMessage,
    so the ReAct loop can react and try something else instead of hanging.
    """
    return await asyncio.wait_for(execute(request), timeout=settings.mcp_tool_call_timeout_seconds)


def make_react_researcher_node(llm_with_tools: BaseChatModel | Runnable) -> Callable:  # type: ignore[type-arg]
    """Return an async react_researcher node with tools already bound to the LLM.

    Args:
        llm_with_tools: LLM with MCP tools bound via llm.bind_tools(mcp_tools).
    """

    @with_dead_letter("react_researcher")
    async def react_researcher(state: "AgentState", config: RunnableConfig) -> dict:
        step = state.get("react_steps", 0) + 1
        logger.info(
            "react_researcher.inputs",
            step=step,
            message_count=len(state["messages"]),
        )
        response = await llm_invoke_with_retry(llm_with_tools, state["messages"], config)  # type: ignore[arg-type]
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls and step >= settings.max_react_steps:
            # Ceiling reached: route_to_writer will skip the tools node for this
            # response, so the tool_calls must be dropped here — otherwise an
            # AIMessage with unresolved tool_calls is persisted to the checkpoint,
            # which providers reject on replay/time-travel.
            logger.warning(
                "react_researcher.ceiling_hit_dropping_tool_calls", step=step, tool_call_count=len(tool_calls)
            )
            response = response.model_copy(update={"tool_calls": [], "invalid_tool_calls": []})
            tool_calls = None
        logger.info("react_researcher.response", step=step, tool_call_count=len(tool_calls) if tool_calls else 0)
        return {"messages": [response], "react_steps": step}

    return react_researcher


def make_react_researcher_node_from_llm(llm: BaseChatModel, mcp_tools: list[BaseTool]) -> Callable:
    """Convenience factory: bind tools to the LLM then return the node.

    Args:
        llm: Base LLM without tools.
        mcp_tools: List of MCP BaseTool instances to bind.
    """
    llm_with_tools = llm.bind_tools(mcp_tools)
    return make_react_researcher_node(llm_with_tools)


def make_tools_node(mcp_tools: list[BaseTool]) -> ToolNode:
    """Return the ToolNode that executes react_researcher's tool_calls, with each
    call bounded by AGENT_MCP_TOOL_CALL_TIMEOUT_SECONDS via awrap_tool_call.

    handle_tool_errors=True is required here: ToolNode's own default only
    auto-converts ToolInvocationError into an error ToolMessage and re-raises
    everything else (including our TimeoutError) uncaught — which would crash
    the whole graph run instead of letting the model see the failure and adjust.
    """
    return ToolNode(mcp_tools, awrap_tool_call=_timeout_bounded_tool_call, handle_tool_errors=True)
