import uuid

from langchain_core.messages import HumanMessage

from app.graph.workflow import compile_simple_graph
from tests.conftest import langgraph_config


class TestSessionContinuation:
    async def test_messages_accumulate_within_session(self, async_checkpointer):
        """The operator.add reducer appends messages across invocations.

        Without the reducer annotation on AgentState.messages, each ainvoke
        would overwrite the list with only the messages produced in that turn.
        This test confirms that the second invocation sees both human messages
        from turn 1 and turn 2 in the accumulated state.
        """
        graph = compile_simple_graph(async_checkpointer)
        tid = f"test-accumulate-{uuid.uuid4()}"

        await graph.ainvoke({"messages": [HumanMessage("Hello!")], "status": "planning"}, langgraph_config(tid))
        result = await graph.ainvoke(
            {"messages": [HumanMessage("How are you?")], "status": "planning"}, langgraph_config(tid)
        )

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 2
        assert human_messages[0].content == "Hello!"
        assert human_messages[1].content == "How are you?"

    async def test_session_persists_across_graph_recompiles(self, async_checkpointer):
        """Checkpoints survive graph object recreation.

        Persistence lives in Postgres, not in the CompiledStateGraph object.
        A new graph instance with the same checkpointer picks up where the
        previous one left off — the in-process graph object is stateless.
        """
        tid = f"test-persist-{uuid.uuid4()}"

        graph1 = compile_simple_graph(async_checkpointer)
        await graph1.ainvoke({"messages": [HumanMessage("First message")], "status": "planning"}, langgraph_config(tid))

        graph2 = compile_simple_graph(async_checkpointer)
        result = await graph2.ainvoke(
            {"messages": [HumanMessage("Second message")], "status": "planning"}, langgraph_config(tid)
        )

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 2

    async def test_checkpoint_count_matches_invocations(self, async_checkpointer):
        """LangGraph writes one checkpoint per node plus one source=input per invoke.

        For this 2-node graph (planner → writer), each ainvoke produces 4 checkpoints:
          source=input  — carry-over state before __start__ processes new input
          source=loop   — after planner completes
          source=loop   — after writer completes
          source=loop   — after __end__ (the final "done" snapshot)
        3 invocations × 4 checkpoints = 12 total in aget_state_history.
        """
        graph = compile_simple_graph(async_checkpointer)
        tid = f"test-count-{uuid.uuid4()}"

        for i in range(3):
            await graph.ainvoke(
                {"messages": [HumanMessage(f"Message {i}")], "status": "planning"}, langgraph_config(tid)
            )

        history = [s async for s in graph.aget_state_history(langgraph_config(tid))]
        assert len(history) == 12
