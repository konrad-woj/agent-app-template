from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import MessagesState, StateGraph

DB_URI = "postgresql://postgres:postgres@localhost:5433/langgraph"  # Mind postgres normally uses 5432 port.


def chat_node(state: MessagesState) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Response to: {last}")]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.set_finish_point("chat")


def main() -> None:
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Create necessary tables if they don't exist (safe to call multiple times)
        checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)

        config: RunnableConfig = {"configurable": {"thread_id": "session-123"}}

        graph.invoke({"messages": [HumanMessage("Hello!")]}, config)
        graph.invoke({"messages": [HumanMessage("How are you?")]}, config)
        graph.invoke({"messages": [HumanMessage("What are you doing?")]}, config)

        history = list(graph.get_state_history(config))
        # LangGraph creates 3 checkpoints per invoke: source=input before __start__,
        # source=loop after node, source=loop after finish point. So 3 invokes = 9 checkpoints.
        print(f"Number of checkpoints: {len(history)}")

    # Example: fetching a previous state and doing REPLAY and FORK from there.
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config: RunnableConfig = {"configurable": {"thread_id": "session-123"}}
        history = list(graph.get_state_history(config))

        one_convo_back = history[2]
        print(f"State from 1 conversation back: {one_convo_back.values['messages']}")
        configurable = one_convo_back.config.get("configurable") or {}
        print(f"Checkpoint ID: {configurable.get('checkpoint_id')}")

        # REPLAY -- re-execute the graph from that checkpoint state, which will yield the same result as before
        # (deterministic in this example since no randomness or external API calls)
        replay_result = graph.invoke(None, one_convo_back.config)
        print(f"Replay — last message: {replay_result['messages'][-1].content}")

        # FORK - execute the graph from that checkpoint state but with a different input,
        # which will yield a different result
        fork_result = graph.invoke({"messages": [HumanMessage("Let's start over!")]}, one_convo_back.config)
        print(f"Fork — last message: {fork_result['messages'][-1].content}")


if __name__ == "__main__":
    main()
