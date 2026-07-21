"""Tests for evals/run_experiment.py's harness-integrity behaviour.

No live server or Postgres involved — `_run_conversation` and `_read_history`
are mocked at their call sites so these exercise pure orchestration logic:
error isolation in `make_task`'s task function, and the one-pool-per-variant
wiring in `_run_variant`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from evals.run_experiment import ExperimentVariant, _run_variant, make_task


class TestMakeTaskErrorIsolation:
    async def test_http_failure_returns_task_error(self) -> None:
        item = {"id": "i1", "input": {"messages": [{"content": "hello"}]}}
        checkpointer = MagicMock(name="checkpointer")
        task = make_task("http://localhost:8000", checkpointer)

        with patch("evals.run_experiment._run_conversation", side_effect=RuntimeError("connection refused")):
            result = await task(item=item)

        assert result["status"] == "task_error"
        assert result["final_answer"] is None
        assert result["history"] == []
        assert result["final_state"] == {}
        assert "connection refused" in result["error"]

    async def test_history_read_failure_preserves_http_response(self) -> None:
        """A DB blip after a successful HTTP call must not discard the model's response."""
        item = {"id": "i1", "input": {"messages": [{"content": "hello"}]}}
        checkpointer = MagicMock(name="checkpointer")
        task = make_task("http://localhost:8000", checkpointer)

        response = {"status": "done", "final_answer": "the answer", "guard_reason": None, "dead_letter": None}
        with (
            patch("evals.run_experiment._run_conversation", AsyncMock(return_value=(response, 1, False))),
            patch("evals.run_experiment._read_history", side_effect=RuntimeError("db unreachable")),
        ):
            result = await task(item=item)

        assert result["status"] == "done"
        assert result["final_answer"] == "the answer"
        assert result["history"] == []
        assert result["final_state"] == {}


class TestRunVariantConnectionPooling:
    async def test_opens_one_pool_and_shares_one_checkpointer_across_items(self) -> None:
        variant = ExperimentVariant(name="default", base_url="http://localhost:8000")
        items = [
            {"id": "i1", "expected_output": {}},
            {"id": "i2", "expected_output": {}},
        ]

        fake_task_output = {"thread_id": "t", "status": "done", "final_answer": "x"}
        fake_task = AsyncMock(return_value=fake_task_output)
        make_task_calls: list[tuple] = []

        def fake_make_task(base_url, checkpointer):
            make_task_calls.append((base_url, checkpointer))
            return fake_task

        pool_instance = MagicMock(name="pool")
        pool_cm = MagicMock(name="pool_context_manager")
        pool_cm.__aenter__ = AsyncMock(return_value=pool_instance)
        pool_cm.__aexit__ = AsyncMock(return_value=False)
        pool_ctor = MagicMock(return_value=pool_cm)

        checkpointer_instances: list[tuple] = []

        def fake_checkpointer_ctor(pool):
            instance = MagicMock(name="checkpointer")
            checkpointer_instances.append((pool, instance))
            return instance

        with (
            patch("evals.run_experiment.make_task", side_effect=fake_make_task),
            patch("psycopg_pool.AsyncConnectionPool", pool_ctor),
            patch("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", side_effect=fake_checkpointer_ctor),
        ):
            result = await _run_variant(
                variant, items, evaluators=[], max_concurrency=2, langfuse=None, db_uri="postgresql://test"
            )

        assert pool_ctor.call_count == 1
        _, kwargs = pool_ctor.call_args
        assert kwargs["conninfo"] == "postgresql://test"
        assert kwargs["max_size"] == 2

        assert len(checkpointer_instances) == 1
        assert checkpointer_instances[0][0] is pool_instance

        assert len(make_task_calls) == 1
        assert make_task_calls[0][0] == "http://localhost:8000"
        assert make_task_calls[0][1] is checkpointer_instances[0][1]

        assert fake_task.call_count == 2
        assert result["name"] == "default"
        assert result["base_url"] == "http://localhost:8000"
