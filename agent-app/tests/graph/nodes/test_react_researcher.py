"""Tests for react_researcher node."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from app.graph.nodes.react_researcher import make_react_researcher_node, make_tools_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


def _build_tools_test_graph(tools_node: ToolNode) -> CompiledStateGraph:
    """Compile a minimal single-node graph so ToolNode gets the runtime context
    (CONFIG_KEY_RUNTIME) it needs — invoking it standalone raises a config error."""
    graph: StateGraph = StateGraph(MessagesState)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile(checkpointer=InMemorySaver())


class TestReactResearcherNode:
    async def test_increments_react_steps(self) -> None:
        llm = make_mock_llm("Thinking...")
        node = make_react_researcher_node(llm)
        result = await node(base_state(react_steps=3), CONFIG)
        assert result["react_steps"] == 4

    async def test_appends_response_to_messages(self) -> None:
        llm = make_mock_llm("I gathered some context.")
        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "I gathered some context."

    async def test_response_with_tool_calls_is_preserved(self) -> None:
        tool_calls = [
            ToolCall(name="web_search", args={"query": "test"}, id="call_1"),
        ]
        response = AIMessage(content="", tool_calls=tool_calls)
        from unittest.mock import AsyncMock, MagicMock

        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(return_value=response)
        llm.bind_tools = MagicMock(return_value=llm)

        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "web_search"
        assert result["messages"][0].tool_calls[0]["args"] == {"query": "test"}

    async def test_dead_letter_on_llm_error(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_react_researcher_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "react_researcher"


class TestMakeToolsNode:
    async def test_fast_tool_call_succeeds(self) -> None:
        @tool
        async def instant_tool(query: str) -> str:
            """An instant tool."""
            return f"result for {query}"

        graph = _build_tools_test_graph(make_tools_node([instant_tool]))
        message = AIMessage(content="", tool_calls=[ToolCall(name="instant_tool", args={"query": "x"}, id="call_1")])
        result = await graph.ainvoke({"messages": [HumanMessage(content="hi"), message]}, CONFIG)
        tool_message = result["messages"][-1]
        assert tool_message.status != "error"
        assert "result for x" in tool_message.content

    async def test_hung_tool_call_times_out_as_error_message(self, monkeypatch) -> None:
        from app.config import settings

        monkeypatch.setattr(settings, "mcp_tool_call_timeout_seconds", 0.05)

        @tool
        async def hung_tool(query: str) -> str:
            """A tool that never returns in time."""
            await asyncio.sleep(1.0)
            return "never gets here"

        graph = _build_tools_test_graph(make_tools_node([hung_tool]))
        message = AIMessage(content="", tool_calls=[ToolCall(name="hung_tool", args={"query": "x"}, id="call_1")])
        result = await graph.ainvoke({"messages": [HumanMessage(content="hi"), message]}, CONFIG)
        tool_message = result["messages"][-1]
        assert tool_message.status == "error"
