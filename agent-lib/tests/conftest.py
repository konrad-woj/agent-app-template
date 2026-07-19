import pytest
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5433/langgraph"


@pytest.fixture(scope="session")
def checkpointer():
    try:
        with PostgresSaver.from_conn_string(DB_URI) as cp:
            cp.setup()
            yield cp
    except Exception as e:
        pytest.skip(f"Postgres unavailable: {e}")
