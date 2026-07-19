"""Router-level tests for /health, /v1/chat, /v1/chat/stream, and /v1/threads/* endpoints.

Uses a standalone FastAPI app (no lifespan) with the graph dependency overridden
by a mock, so no Postgres or MCP server is required.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from app.dependencies import get_graph
from app.exceptions import LLMRateLimitError
from app.graph.nodes._dead_letter import dead_letter_counter
from app.routers import _classify_error, _generate, health_router, router

_app = FastAPI()
_app.include_router(health_router)
_app.include_router(router)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> MagicMock:
    g = MagicMock()
    g.aget_state = AsyncMock()
    g.ainvoke = AsyncMock()
    g.aupdate_state = AsyncMock()
    return g


@pytest.fixture
async def client(mock_graph: MagicMock) -> AsyncGenerator[AsyncClient]:
    _app.dependency_overrides[get_graph] = lambda: mock_graph
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    next_nodes: tuple = (),
    values: dict | None = None,
    tasks: list | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    s = MagicMock()
    s.next = next_nodes
    s.values = values if values is not None else {}
    s.tasks = tasks or []
    s.config = config or {"configurable": {"checkpoint_id": "cp-1"}}
    s.metadata = metadata or {"step": 0, "source": "loop"}
    return s


def _interrupted_snapshot(plan_value: dict | None = None) -> MagicMock:
    raw = MagicMock()
    raw.value = plan_value or {"plan": ["Step 1", "Step 2"]}
    task = MagicMock()
    task.interrupts = [raw]
    return _snapshot(
        next_nodes=("planner",),
        values={"status": "planning"},
        tasks=[task],
    )


async def _async_history(*snaps: MagicMock):
    for s in snaps:
        yield s


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_ok(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /metrics/dead-letter
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_dead_letter_counter_for_router_tests():
    dead_letter_counter._by_node.clear()
    yield
    dead_letter_counter._by_node.clear()


class TestDeadLetterMetrics:
    async def test_returns_zero_counts_when_nothing_dead_lettered(self, client: AsyncClient) -> None:
        response = await client.get("/metrics/dead-letter")
        assert response.status_code == 200
        assert response.json() == {"total": 0, "by_node": {}}

    async def test_reflects_counter_state(self, client: AsyncClient) -> None:
        dead_letter_counter.increment("planner")
        dead_letter_counter.increment("planner")
        dead_letter_counter.increment("writer")

        response = await client.get("/metrics/dead-letter")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["by_node"] == {"planner": 2, "writer": 1}


# ---------------------------------------------------------------------------
# GET /ready
# ---------------------------------------------------------------------------


@pytest.fixture
def app_state() -> Generator[Any]:
    """Set up a fully-healthy app.state, torn down after the test."""
    gliguard = MagicMock()
    gliguard.loaded = True
    checkpointer = MagicMock()
    pool_conn = AsyncMock()
    checkpointer.conn.connection.return_value.__aenter__ = AsyncMock(return_value=pool_conn)
    checkpointer.conn.connection.return_value.__aexit__ = AsyncMock(return_value=False)
    _app.state.gliguard = gliguard
    _app.state.checkpointer = checkpointer
    _app.state.mcp_tool_count = 3
    yield _app.state
    for attr in ("gliguard", "checkpointer", "mcp_tool_count"):
        if hasattr(_app.state, attr):
            delattr(_app.state, attr)


class TestReady:
    async def test_all_dependencies_healthy_returns_200(self, client: AsyncClient, app_state: MagicMock) -> None:
        response = await client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["checks"] == {"gliguard_loaded": True, "database": True, "mcp_tools_loaded": True}

    async def test_gliguard_not_loaded_returns_503(self, client: AsyncClient, app_state: MagicMock) -> None:
        app_state.gliguard.loaded = False
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["gliguard_loaded"] is False

    async def test_database_unreachable_returns_503(self, client: AsyncClient, app_state: MagicMock) -> None:
        app_state.checkpointer.conn.connection.return_value.__aenter__ = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["database"] is False

    async def test_no_mcp_tools_returns_503(self, client: AsyncClient, app_state: MagicMock) -> None:
        app_state.mcp_tool_count = 0
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["mcp_tools_loaded"] is False

    async def test_missing_dependencies_returns_503(self, client: AsyncClient) -> None:
        """Before lifespan populates app.state (or if a dependency was never set)."""
        response = await client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["checks"] == {"gliguard_loaded": False, "database": False, "mcp_tools_loaded": False}


# ---------------------------------------------------------------------------
# POST /v1/chat — fresh turn (not interrupted)
# ---------------------------------------------------------------------------


class TestChatFreshTurn:
    async def test_invokes_and_returns_result(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Research result."})

        response = await client.post("/v1/chat", json={"thread_id": "t-1", "message": "Research AI"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["final_answer"] == "Research result."
        assert data["is_interrupted"] is False

    async def test_thread_id_passed_in_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done"})

        await client.post("/v1/chat", json={"thread_id": "my-thread", "message": "hi"})

        config = mock_graph.aget_state.call_args_list[0][0][0]
        assert config["configurable"]["thread_id"] == "my-thread"

    async def test_guard_blocked_propagates_reason(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "blocked", "guard_reason": "Off-topic request."})

        response = await client.post("/v1/chat", json={"thread_id": "t-2", "message": "bad"})

        data = response.json()
        assert data["status"] == "blocked"
        assert data["guard_reason"] == "Off-topic request."

    async def test_new_interrupt_after_invocation_sets_is_interrupted(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """Graph suspends mid-run (plan_review fires interrupt); response reflects the new interrupt."""
        mock_graph.aget_state = AsyncMock(side_effect=[_snapshot(), _interrupted_snapshot({"plan": ["Step A"]})])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "planning"})

        response = await client.post("/v1/chat", json={"thread_id": "t-3", "message": "Research topic"})

        data = response.json()
        assert data["is_interrupted"] is True
        assert data["interrupt_value"] == {"plan": ["Step A"]}

    async def test_missing_status_defaults_to_done(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={})

        response = await client.post("/v1/chat", json={"thread_id": "t-4", "message": "hello"})

        assert response.json()["status"] == "done"

    async def test_dead_letter_info_surfaced_in_chat_response(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        dead_letter_info = {"failed_node": "input_guard", "error_type": "RuntimeError", "error_message": "boom"}
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(return_value={"status": "dead_lettered", "dead_letter": dead_letter_info})

        response = await client.post("/v1/chat", json={"thread_id": "t-dl", "message": "hello"})

        data = response.json()
        assert data["status"] == "dead_lettered"
        assert data["dead_letter"] == dead_letter_info


class TestChatErrorClassification:
    """/v1/chat must classify LLM/DB errors the same way /v1/chat/stream does via
    _classify_error, instead of letting them fall through to a generic 500."""

    async def test_rate_limit_error_returns_429(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        from app.exceptions import LLMRateLimitError

        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(side_effect=LLMRateLimitError("too many requests"))

        response = await client.post("/v1/chat", json={"thread_id": "t-err-1", "message": "hi"})

        assert response.status_code == 429
        assert response.json()["detail"] == "LLM rate limit exceeded"

    async def test_service_unavailable_error_returns_503(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        from app.exceptions import LLMServiceUnavailableError

        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("provider down"))

        response = await client.post("/v1/chat", json={"thread_id": "t-err-2", "message": "hi"})

        assert response.status_code == 503
        assert response.json()["detail"] == "LLM service unavailable"

    async def test_service_error_returns_502(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        from app.exceptions import LLMServiceError

        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(side_effect=LLMServiceError("unexpected provider error"))

        response = await client.post("/v1/chat", json={"thread_id": "t-err-3", "message": "hi"})

        assert response.status_code == 502
        assert response.json()["detail"] == "LLM service error"

    async def test_recursion_error_returns_500_with_specific_detail(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        from langgraph.errors import GraphRecursionError

        mock_graph.aget_state = AsyncMock(return_value=_snapshot())
        mock_graph.ainvoke = AsyncMock(side_effect=GraphRecursionError())

        response = await client.post("/v1/chat", json={"thread_id": "t-err-4", "message": "hi"})

        assert response.status_code == 500
        assert response.json()["detail"] == "Pipeline step limit exceeded"

    async def test_initial_state_lookup_failure_is_classified(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        """A DB error on the pre-invoke aget_state call must also be classified, not just
        errors from ainvoke — both hit the checkpointer and can fail the same way."""
        from app.exceptions import LLMServiceUnavailableError

        mock_graph.aget_state = AsyncMock(side_effect=LLMServiceUnavailableError("db down"))

        response = await client.post("/v1/chat", json={"thread_id": "t-err-5", "message": "hi"})

        assert response.status_code == 503
        assert response.json()["detail"] == "LLM service unavailable"
        mock_graph.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# POST /v1/chat — interrupted thread
# ---------------------------------------------------------------------------


class TestChatInterruptedThread:
    async def test_approve_none_returns_interrupted_without_invoking(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_interrupted_snapshot())

        response = await client.post("/v1/chat", json={"thread_id": "t-5", "message": "whatever"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"
        assert data["is_interrupted"] is True
        mock_graph.ainvoke.assert_not_called()

    async def test_approve_none_returns_interrupt_value(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_interrupted_snapshot({"plan": ["Search for X", "Analyse Y"]}))

        response = await client.post("/v1/chat", json={"thread_id": "t-6", "message": "hi"})

        assert response.json()["interrupt_value"] == {"plan": ["Search for X", "Analyse Y"]}

    async def test_approve_none_with_no_tasks_returns_null_interrupt_value(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        snap = _snapshot(next_nodes=("planner",), values={"status": "planning"}, tasks=[])
        mock_graph.aget_state = AsyncMock(return_value=snap)

        response = await client.post("/v1/chat", json={"thread_id": "t-7", "message": "hi"})

        assert response.json()["interrupt_value"] is None

    async def test_approve_true_resumes_with_command(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(side_effect=[_interrupted_snapshot(), _snapshot()])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Answer"})

        response = await client.post("/v1/chat", json={"thread_id": "t-8", "message": "hi", "approve": True})

        assert response.status_code == 200
        invoked_input = mock_graph.ainvoke.call_args[0][0]
        assert isinstance(invoked_input, Command)
        assert invoked_input.resume is True

    async def test_approve_false_resumes_with_reject_command(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(side_effect=[_interrupted_snapshot(), _snapshot()])
        mock_graph.ainvoke = AsyncMock(return_value={"status": "aborted"})

        response = await client.post("/v1/chat", json={"thread_id": "t-9", "message": "hi", "approve": False})

        assert response.status_code == 200
        invoked_input = mock_graph.ainvoke.call_args[0][0]
        assert isinstance(invoked_input, Command)
        assert invoked_input.resume is False


# ---------------------------------------------------------------------------
# POST /v1/chat — request validation
# ---------------------------------------------------------------------------


class TestChatValidation:
    async def test_blank_message_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "t-1", "message": "   "})
        assert response.status_code == 422

    async def test_blank_thread_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "   ", "message": "hello"})
        assert response.status_code == 422

    async def test_missing_message_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"thread_id": "t-1"})
        assert response.status_code == 422

    async def test_missing_thread_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/chat", json={"message": "hello"})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/threads/{thread_id}/history
# ---------------------------------------------------------------------------


class TestHistory:
    async def test_empty_history_returns_empty_list(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state_history = MagicMock(return_value=_async_history())

        response = await client.get("/v1/threads/t-1/history")

        assert response.status_code == 200
        assert response.json() == []

    async def test_returns_correct_checkpoint_info_shape(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        snap = _snapshot(
            next_nodes=("writer",),
            values={"status": "writing", "messages": [HumanMessage("hi")]},
            config={"configurable": {"checkpoint_id": "cp-abc"}},
            metadata={"step": 3, "source": "loop"},
        )
        mock_graph.aget_state_history = MagicMock(return_value=_async_history(snap))

        response = await client.get("/v1/threads/t-1/history")

        assert response.status_code == 200
        items = response.json()
        assert len(items) == 1
        item = items[0]
        assert item["checkpoint_id"] == "cp-abc"
        assert item["step"] == 3
        assert item["source"] == "loop"
        assert item["next"] == ["writer"]
        assert item["status"] == "writing"
        assert item["messages_count"] == 1

    async def test_returns_all_snapshots_in_order(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        snaps = [
            _snapshot(
                config={"configurable": {"checkpoint_id": f"cp-{i}"}},
                metadata={"step": i, "source": "loop"},
            )
            for i in range(3)
        ]
        mock_graph.aget_state_history = MagicMock(return_value=_async_history(*snaps))

        response = await client.get("/v1/threads/t-1/history")

        items = response.json()
        assert len(items) == 3
        assert [item["checkpoint_id"] for item in items] == ["cp-0", "cp-1", "cp-2"]

    async def test_passes_thread_id_in_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state_history = MagicMock(return_value=_async_history())

        await client.get("/v1/threads/specific-thread/history")

        config = mock_graph.aget_state_history.call_args[0][0]
        assert config["configurable"]["thread_id"] == "specific-thread"


# ---------------------------------------------------------------------------
# POST /v1/threads/{thread_id}/replay
# ---------------------------------------------------------------------------


class TestReplay:
    async def test_invokes_with_none_input_and_correct_config(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "Replayed."})

        await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "cp-xyz"})

        args, _ = mock_graph.ainvoke.call_args
        assert args[0] is None  # None = replay from stored state, no new input
        assert args[1]["configurable"]["thread_id"] == "t-1"
        assert args[1]["configurable"]["checkpoint_id"] == "cp-xyz"

    async def test_returns_result_with_correct_thread_id(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.ainvoke = AsyncMock(return_value={"status": "done", "final_answer": "The replayed answer."})

        response = await client.post("/v1/threads/t-2/replay", json={"checkpoint_id": "cp-1"})

        data = response.json()
        assert data["thread_id"] == "t-2"
        assert data["status"] == "done"
        assert data["final_answer"] == "The replayed answer."

    async def test_missing_checkpoint_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/threads/t-1/replay", json={})
        assert response.status_code == 422

    async def test_blank_checkpoint_id_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "   "})
        assert response.status_code == 422

    async def test_invalid_checkpoint_id_returns_404(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        """Postgres raises on a malformed checkpoint_id UUID during the state lookup."""
        mock_graph.aget_state = AsyncMock(side_effect=Exception("invalid input syntax for type uuid"))

        response = await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "not-a-uuid"})

        assert response.status_code == 404
        assert "not-a-uuid" in response.json()["detail"]
        mock_graph.ainvoke.assert_not_called()

    async def test_no_matching_checkpoint_returns_404(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        """A well-formed but nonexistent checkpoint_id returns an empty snapshot, not an exception."""
        mock_graph.aget_state = AsyncMock(return_value=_snapshot(values={}))

        response = await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "cp-missing"})

        assert response.status_code == 404
        assert "cp-missing" in response.json()["detail"]
        mock_graph.ainvoke.assert_not_called()

    async def test_execution_failure_during_replay_is_classified_not_reported_as_404(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        """A real failure while re-running the graph (e.g. LLM outage) must not be misreported
        as "checkpoint not found" — that would hide a genuine incident from 5xx-based alerting."""
        from app.exceptions import LLMServiceUnavailableError

        mock_graph.aget_state = AsyncMock(return_value=_snapshot(values={"status": "planning"}))
        mock_graph.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("provider down"))

        response = await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "cp-1"})

        assert response.status_code == 503
        assert response.json()["detail"] == "LLM service unavailable"

    async def test_dead_letter_info_surfaced_in_response(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        dead_letter_info = {"failed_node": "input_guard", "error_type": "RuntimeError", "error_message": "boom"}
        mock_graph.ainvoke = AsyncMock(return_value={"status": "dead_lettered", "dead_letter": dead_letter_info})

        response = await client.post("/v1/threads/t-1/replay", json={"checkpoint_id": "cp-1"})

        data = response.json()
        assert data["status"] == "dead_lettered"
        assert data["dead_letter"] == dead_letter_info


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of {"event": ..., "data": ...} dicts."""
    frames = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                import json

                data = json.loads(line[5:].strip())
        if event_type is not None:
            frames.append({"event": event_type, "data": data})
    return frames


def _make_stream_event(content: str, node: str) -> dict:
    chunk = AIMessageChunk(content=content)
    return {"event": "on_chat_model_stream", "metadata": {"langgraph_node": node}, "data": {"chunk": chunk}}


def _astream(*events):
    """Return an async generator of events — use as astream_events return value."""

    async def _gen(*args, **kwargs):
        for e in events:
            yield e

    return _gen()


# ---------------------------------------------------------------------------
# POST /v1/chat/stream
# ---------------------------------------------------------------------------


class TestChatStream:
    async def test_stream_emits_token_frames(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),  # pre-invoke check
                _snapshot(values={"status": "done", "final_answer": "Result"}),  # post-stream check
            ]
        )
        events = [
            _make_stream_event("The", "writer"),
            _make_stream_event(" answer", "writer"),
            _make_stream_event(" is here", "writer"),
        ]
        mock_graph.astream_events = MagicMock(return_value=_astream(*events))

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s1", "message": "Research AI"})

        frames = _parse_sse(response.text)
        token_frames = [f for f in frames if f["event"] == "token"]
        assert len(token_frames) == 3
        assert token_frames[0]["data"]["token"] == "The"
        assert token_frames[1]["data"]["token"] == " answer"
        assert token_frames[2]["data"]["token"] == " is here"
        done_frames = [f for f in frames if f["event"] == "done"]
        assert len(done_frames) == 1

    async def test_stream_interrupt_frame(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),  # pre-invoke: not interrupted
                _interrupted_snapshot({"plan": ["Step A", "Step B"]}),  # post-stream: interrupted
            ]
        )
        mock_graph.astream_events = MagicMock(return_value=_astream())

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s2", "message": "Research topic"})

        frames = _parse_sse(response.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "interrupt"
        assert frames[0]["data"]["interrupt_value"] == {"plan": ["Step A", "Step B"]}

    async def test_stream_approve_none_on_interrupted_thread_emits_interrupt_frame(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_interrupted_snapshot({"plan": ["Step 1"]}))

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s3", "message": "anything"})

        frames = _parse_sse(response.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "interrupt"
        assert frames[0]["data"]["interrupt_value"] == {"plan": ["Step 1"]}
        mock_graph.astream_events.assert_not_called()

    async def test_stream_resume_via_stream_endpoint(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _interrupted_snapshot(),  # pre-invoke: interrupted
                _snapshot(values={"status": "done", "final_answer": "Done"}),  # post-stream
            ]
        )
        mock_graph.astream_events = MagicMock(return_value=_astream(_make_stream_event("Done", "writer")))

        response = await client.post(
            "/v1/chat/stream", json={"thread_id": "t-s4", "message": "approve", "approve": True}
        )

        frames = _parse_sse(response.text)
        token_frames = [f for f in frames if f["event"] == "token"]
        assert len(token_frames) == 1
        done_frames = [f for f in frames if f["event"] == "done"]
        assert len(done_frames) == 1
        assert done_frames[0]["data"]["status"] == "done"

        invoked_input = mock_graph.astream_events.call_args[0][0]
        assert isinstance(invoked_input, Command)
        assert invoked_input.resume is True

    async def test_stream_done_frame_carries_status_and_answer(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),
                _snapshot(values={"status": "done", "final_answer": "The final answer"}),
            ]
        )
        mock_graph.astream_events = MagicMock(return_value=_astream())

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s5", "message": "Research"})

        frames = _parse_sse(response.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "done"
        assert frames[0]["data"]["status"] == "done"
        assert frames[0]["data"]["final_answer"] == "The final answer"

    async def test_stream_error_frame_on_rate_limit(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(return_value=_snapshot())

        async def _failing_stream(*args, **kwargs):
            raise LLMRateLimitError("rate limited")
            yield  # make it an async generator

        mock_graph.astream_events = MagicMock(return_value=_failing_stream())

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s6", "message": "Research"})

        frames = _parse_sse(response.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "error"
        assert frames[0]["data"]["code"] == 429
        assert frames[0]["data"]["detail"] == "LLM rate limit exceeded"

    async def test_stream_no_tokens_from_guard_nodes(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),
                _snapshot(values={"status": "done", "final_answer": "ok"}),
            ]
        )
        events = [
            _make_stream_event("guard token", "input_guard"),
            _make_stream_event("planner token", "planner"),
            _make_stream_event("writer token", "writer"),
            _make_stream_event("output token", "output_guard"),
        ]
        mock_graph.astream_events = MagicMock(return_value=_astream(*events))

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s7", "message": "Research"})

        frames = _parse_sse(response.text)
        token_frames = [f for f in frames if f["event"] == "token"]
        assert len(token_frames) == 1
        assert token_frames[0]["data"]["token"] == "writer token"

    async def test_stream_thinking_mode_list_content_emits_text_only(
        self, client: AsyncClient, mock_graph: MagicMock
    ) -> None:
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),
                _snapshot(values={"status": "done", "final_answer": "Result"}),
            ]
        )
        thinking_chunk = AIMessageChunk(content=[{"type": "thinking", "thinking": "internal reasoning"}])
        text_chunk = AIMessageChunk(content=[{"type": "text", "text": "Final answer"}])
        events = [
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "writer"},
                "data": {"chunk": thinking_chunk},
            },
            {"event": "on_chat_model_stream", "metadata": {"langgraph_node": "writer"}, "data": {"chunk": text_chunk}},
        ]
        mock_graph.astream_events = MagicMock(return_value=_astream(*events))

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s9", "message": "Research"})

        frames = _parse_sse(response.text)
        token_frames = [f for f in frames if f["event"] == "token"]
        assert len(token_frames) == 1
        assert token_frames[0]["data"]["token"] == "Final answer"

    async def test_stream_dead_lettered_arrives_as_error(self, client: AsyncClient, mock_graph: MagicMock) -> None:
        dead_letter_info = {"failed_node": "input_guard", "error_type": "RuntimeError", "error_message": "boom"}
        mock_graph.aget_state = AsyncMock(
            side_effect=[
                _snapshot(),
                _snapshot(values={"status": "dead_lettered", "dead_letter": dead_letter_info}),
            ]
        )
        mock_graph.astream_events = MagicMock(return_value=_astream())

        response = await client.post("/v1/chat/stream", json={"thread_id": "t-s8", "message": "Research"})

        frames = _parse_sse(response.text)
        assert len(frames) == 1
        assert frames[0]["event"] == "error"
        assert frames[0]["data"]["status"] == "dead_lettered"
        assert frames[0]["data"]["dead_letter"] == dead_letter_info

    async def test_stream_disconnect_stops_generator(self) -> None:
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=_snapshot(values={"status": "done"}))

        events = [_make_stream_event(f"token {i}", "writer") for i in range(10)]
        mock_graph.astream_events = MagicMock(return_value=_astream(*events))

        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=True)

        from langchain_core.runnables import RunnableConfig

        config: RunnableConfig = {"configurable": {"thread_id": "t-disc"}, "recursion_limit": 50}
        graph_input = {"messages": [HumanMessage(content="test")], "status": "planning"}

        collected = []
        async for frame in _generate(mock_graph, graph_input, config, mock_request):
            collected.append(frame)

        assert collected == []


# ---------------------------------------------------------------------------
# _classify_error unit tests
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_rate_limit_error(self) -> None:
        from app.exceptions import LLMRateLimitError

        code, detail = _classify_error(LLMRateLimitError("x"))
        assert code == 429
        assert detail == "LLM rate limit exceeded"

    def test_service_unavailable_error(self) -> None:
        from app.exceptions import LLMServiceUnavailableError

        code, detail = _classify_error(LLMServiceUnavailableError("x"))
        assert code == 503
        assert detail == "LLM service unavailable"

    def test_service_error(self) -> None:
        from app.exceptions import LLMServiceError

        code, detail = _classify_error(LLMServiceError("x"))
        assert code == 502
        assert detail == "LLM service error"

    def test_graph_recursion_error(self) -> None:
        from langgraph.errors import GraphRecursionError

        code, detail = _classify_error(GraphRecursionError())
        assert code == 500
        assert detail == "Pipeline step limit exceeded"

    def test_unknown_error(self) -> None:
        code, detail = _classify_error(RuntimeError("something"))
        assert code == 500
        assert detail == "Internal server error"
