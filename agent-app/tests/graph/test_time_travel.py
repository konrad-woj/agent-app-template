import uuid

from langchain_core.messages import HumanMessage

from app.graph.workflow import compile_simple_graph
from tests.conftest import langgraph_config


def _find_pending_planner_checkpoint(history: list, n_messages: int):
    """Return the source=loop checkpoint with n messages and next=('planner',).

    This is the state where __start__ has appended the new human message but
    the planner has not yet run — the right point for a pure replay.
    """
    return next(
        (
            s
            for s in history
            if s.metadata.get("source") == "loop" and s.next == ("planner",) and len(s.values["messages"]) == n_messages
        ),
        None,
    )


def _find_input_checkpoint_with_n_messages(history: list, n_messages: int):
    """Return the source=input checkpoint with n messages and next=('__start__',).

    source=input checkpoints are the carry-over state at the start of a new
    invoke (before __start__ appends the new message) — the right point for a
    fork so that an injected message becomes the new branch.
    """
    return next(
        (
            s
            for s in history
            if s.metadata.get("source") == "input"
            and s.next == ("__start__",)
            and len(s.values["messages"]) == n_messages
        ),
        None,
    )


class TestTimeTravel:
    async def test_replay_yields_same_result(self, async_checkpointer):
        """ainvoke(None, checkpoint_config) re-executes from a past snapshot.

        Passing None as the input signals LangGraph to restore the stored state
        at the given checkpoint and run forward without injecting new input.
        The graph continues from the node listed in snapshot.next, so using a
        source=loop checkpoint where next=("planner",) replays the turn that
        originally produced that result.
        """
        graph = compile_simple_graph(async_checkpointer)
        tid = f"test-replay-{uuid.uuid4()}"

        await graph.ainvoke({"messages": [HumanMessage("Hello!")], "status": "planning"}, langgraph_config(tid))
        original_result = await graph.ainvoke(
            {"messages": [HumanMessage("Research AI trends")], "status": "planning"}, langgraph_config(tid)
        )
        original_last = original_result["messages"][-1].content

        await graph.ainvoke(
            {"messages": [HumanMessage("Another question")], "status": "planning"}, langgraph_config(tid)
        )

        history = [s async for s in graph.aget_state_history(langgraph_config(tid))]
        # After invoke 2 the accumulated messages are:
        #   [H0, AI_planner0, AI_writer0, H1, AI_planner1, AI_writer1, H2, AI_planner2, AI_writer2]
        # The source=loop checkpoint where next=("planner",) and messages count = 4
        # is the state just after __start__ appended H1 but before planner ran — the
        # right entry point to replay turn 2 and get the same writer output.
        checkpoint = _find_pending_planner_checkpoint(history, 4)
        assert checkpoint is not None, "Could not find pending-planner checkpoint with 4 messages"

        replay_result = await graph.ainvoke(None, checkpoint.config)
        assert replay_result["messages"][-1].content == original_last

    async def test_fork_produces_different_branch(self, async_checkpointer):
        """Supplying new input at a historical checkpoint creates an independent fork.

        A fork injects new state on top of a past snapshot.  The new branch
        diverges from that point and gets its own checkpoint chain within the
        same thread_id.  The source=input checkpoint at the start of an invoke
        (before __start__ appends the new message) is the right fork point —
        the injected message becomes the first new message on the branch.
        """
        graph = compile_simple_graph(async_checkpointer)
        tid = f"test-fork-{uuid.uuid4()}"

        await graph.ainvoke({"messages": [HumanMessage("Hello!")], "status": "planning"}, langgraph_config(tid))
        await graph.ainvoke({"messages": [HumanMessage("Research AI")], "status": "planning"}, langgraph_config(tid))
        await graph.ainvoke({"messages": [HumanMessage("More info")], "status": "planning"}, langgraph_config(tid))

        history = [s async for s in graph.aget_state_history(langgraph_config(tid))]
        # After invoke 0: state = [H0, AI_planner0, AI_writer0] = 3 messages.
        # The source=input checkpoint at the start of invoke 1 carries those 3
        # messages as its snapshot — that is the fork point after the 1st turn.
        checkpoint_after_1st = _find_input_checkpoint_with_n_messages(history, 3)
        assert checkpoint_after_1st is not None, "Could not find source=input checkpoint with 3 messages"

        fork_result = await graph.ainvoke(
            {"messages": [HumanMessage("Let's start over!")], "status": "planning"},
            checkpoint_after_1st.config,
        )

        fork_human = [m for m in fork_result["messages"] if isinstance(m, HumanMessage)]
        assert any(m.content == "Let's start over!" for m in fork_human)

    async def test_fork_does_not_alter_original_thread(self, async_checkpointer):
        """A fork is additive: the original branch remains intact in checkpoint history.

        LangGraph stores all branches under the same thread_id.
        aget_state_history returns checkpoints from every branch, so messages
        from the original turns are still visible after a fork is created.
        """
        graph = compile_simple_graph(async_checkpointer)
        tid = f"test-fork-isolation-{uuid.uuid4()}"

        await graph.ainvoke({"messages": [HumanMessage("Hello!")], "status": "planning"}, langgraph_config(tid))
        await graph.ainvoke({"messages": [HumanMessage("Second message")], "status": "planning"}, langgraph_config(tid))

        history = [s async for s in graph.aget_state_history(langgraph_config(tid))]
        checkpoint_after_1st = _find_input_checkpoint_with_n_messages(history, 3)
        assert checkpoint_after_1st is not None

        await graph.ainvoke(
            {"messages": [HumanMessage("Fork branch")], "status": "planning"},
            checkpoint_after_1st.config,
        )

        updated_history = [s async for s in graph.aget_state_history(langgraph_config(tid))]
        all_human_contents = [
            m.content for state in updated_history for m in state.values["messages"] if isinstance(m, HumanMessage)
        ]
        assert "Second message" in all_human_contents
