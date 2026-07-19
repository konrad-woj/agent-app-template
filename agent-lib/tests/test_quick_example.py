import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_lib.quick_example import builder


def _compile(checkpointer):
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _find_pending_chat_checkpoint(history: list, n: int):
    """Return the source=loop checkpoint with n messages and next=('chat',).

    These are states where __start__ has already appended the new human message
    and the chat node is about to run — the right point for replay with invoke(None).
    """
    return next(
        (
            s
            for s in history
            if s.metadata.get("source") == "loop" and s.next == ("chat",) and len(s.values["messages"]) == n
        ),
        None,
    )


def _find_input_checkpoint_with_n_messages(history: list, n: int):
    """Return the source=input checkpoint with n messages and next=('__start__',).

    source=input checkpoints represent the carry-over state at the start of a new
    invoke (before __start__ appends the new message) — the right point for fork
    so that an injected message becomes the new branch.
    """
    return next(
        (
            s
            for s in history
            if s.metadata.get("source") == "input" and s.next == ("__start__",) and len(s.values["messages"]) == n
        ),
        None,
    )


class TestSessionContinuation:
    def test_messages_accumulate_within_session(self, checkpointer):
        graph = _compile(checkpointer)
        tid = f"test-session-{uuid.uuid4()}"
        config = _config(tid)

        graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        result = graph.invoke({"messages": [HumanMessage("How are you?")]}, config)

        messages = result["messages"]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 2
        assert human_messages[0].content == "Hello!"
        assert human_messages[1].content == "How are you?"

    def test_session_persists_across_graph_recompiles(self, checkpointer):
        tid = f"test-persist-{uuid.uuid4()}"
        config = _config(tid)

        graph1 = _compile(checkpointer)
        graph1.invoke({"messages": [HumanMessage("First message")]}, config)

        graph2 = _compile(checkpointer)
        result = graph2.invoke({"messages": [HumanMessage("Second message")]}, config)

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 2

    def test_checkpoint_count_matches_invocations(self, checkpointer):
        graph = _compile(checkpointer)
        tid = f"test-count-{uuid.uuid4()}"
        config = _config(tid)

        for i in range(3):
            graph.invoke({"messages": [HumanMessage(f"Message {i}")]}, config)

        history = list(graph.get_state_history(config))
        # LangGraph creates 3 checkpoints per invoke (input + after node + after finish point)
        assert len(history) == 9


class TestTimeTravel:
    def test_replay_yields_same_result(self, checkpointer):
        graph = _compile(checkpointer)
        tid = f"test-replay-{uuid.uuid4()}"
        config = _config(tid)

        graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        original_result = graph.invoke({"messages": [HumanMessage("How are you?")]}, config)
        original_last = original_result["messages"][-1].content
        assert original_last == "Response to: How are you?"

        graph.invoke({"messages": [HumanMessage("What are you doing?")]}, config)

        history = list(graph.get_state_history(config))
        # source=loop + next=('chat',) + 3 messages = state after __start__ appended H1,
        # chat node has not yet run — replaying with None re-executes chat on H1.
        checkpoint = _find_pending_chat_checkpoint(history, 3)
        assert checkpoint is not None, "Could not find pending-chat checkpoint with 3 messages"

        replay_result = graph.invoke(None, checkpoint.config)
        assert replay_result["messages"][-1].content == original_last

    def test_fork_produces_different_branch(self, checkpointer):
        graph = _compile(checkpointer)
        tid = f"test-fork-{uuid.uuid4()}"
        config = _config(tid)

        graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        graph.invoke({"messages": [HumanMessage("How are you?")]}, config)
        graph.invoke({"messages": [HumanMessage("What are you doing?")]}, config)

        history = list(graph.get_state_history(config))
        # source=input + 2 messages = carry-over state after invoke 1 finished (H0, A0)
        # and before invoke 2 started — injecting a different message here creates a branch.
        checkpoint_after_1st = _find_input_checkpoint_with_n_messages(history, 2)
        assert checkpoint_after_1st is not None, "Could not find source=input checkpoint with 2 messages"

        fork_result = graph.invoke(
            {"messages": [HumanMessage("Let's start over!")]},
            checkpoint_after_1st.config,
        )

        fork_human = [m for m in fork_result["messages"] if isinstance(m, HumanMessage)]
        assert any(m.content == "Let's start over!" for m in fork_human)
        assert fork_result["messages"][-1].content == "Response to: Let's start over!"

    def test_fork_does_not_alter_original_thread(self, checkpointer):
        graph = _compile(checkpointer)
        tid = f"test-fork-isolation-{uuid.uuid4()}"
        config = _config(tid)

        graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        graph.invoke({"messages": [HumanMessage("Second message")]}, config)

        history = list(graph.get_state_history(config))
        checkpoint_after_1st = _find_input_checkpoint_with_n_messages(history, 2)
        assert checkpoint_after_1st is not None

        graph.invoke(
            {"messages": [HumanMessage("Fork branch")]},
            checkpoint_after_1st.config,
        )

        # Original thread still has the "Second message" branch in history
        updated_history = list(graph.get_state_history(config))
        all_human_contents = [
            m.content for state in updated_history for m in state.values["messages"] if isinstance(m, HumanMessage)
        ]
        assert "Second message" in all_human_contents
