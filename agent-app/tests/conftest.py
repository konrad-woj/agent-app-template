import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5433/langgraph"


@pytest.fixture(scope="session")
async def async_checkpointer():
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            yield checkpointer
    except Exception as e:
        pytest.skip(f"Postgres unavailable: {e}")


def langgraph_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}
