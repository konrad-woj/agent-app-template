"""Dead-letter pattern for graph nodes.

Any unhandled exception in a decorated node is caught, serialised into a
``DeadLetterInfo`` record, written to state, and the graph is routed to the
terminal ``dead_letter_node`` via the ``after()`` routing helper.

This is analogous to a dead-letter queue in messaging: the execution does not
crash or silently disappear — it lands in an observable, structured record that
can be inspected via checkpoint history or replayed via time-travel.

Usage:
    @with_dead_letter("my_node")
    async def my_node(state: AgentState, config: RunnableConfig) -> dict:
        ...

    # In workflow.py, replace add_edge with:
    graph.add_conditional_edges("my_node", after("next_node"))
"""

import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from logger import get_logger

if TYPE_CHECKING:
    from app.graph.state import AgentState

logger = get_logger(__name__)


class DeadLetterInfo(TypedDict):
    failed_node: str
    error_type: str
    error_message: str
    traceback: str
    timestamp: str  # ISO-8601


class DeadLetterCounter:
    """Thread-safe, in-process count of dead-lettered runs, by failed node.

    Not a substitute for a real metrics backend (Prometheus, etc.) — this app has
    none, and the count resets on restart / isn't aggregated across replicas. It
    exists so a log-based alert (CloudWatch Logs Insights, Datadog log metrics,
    Loki) can key off a stable numeric field instead of grepping error strings,
    and so a single long-lived pod's dead-letter rate is visible without one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_node: Counter[str] = Counter()

    def increment(self, node_name: str) -> int:
        """Record one dead-lettered run for `node_name`; return the new total across all nodes."""
        with self._lock:
            self._by_node[node_name] += 1
            return self.total

    @property
    def total(self) -> int:
        return sum(self._by_node.values())

    def snapshot(self) -> dict[str, int]:
        """Return a copy of the per-node counts (safe to log or serve from /ready-style endpoints)."""
        with self._lock:
            return dict(self._by_node)


dead_letter_counter = DeadLetterCounter()


def with_dead_letter(node_name: str) -> Callable:
    """Decorator that catches unhandled exceptions and writes DeadLetterInfo to state.

    Args:
        node_name: The name of the node being decorated, included in the record for traceability.
    """

    def decorator(fn: Callable) -> Callable:
        async def wrapper(state: dict, config: RunnableConfig) -> dict:
            thread_id = (config.get("configurable") or {}).get("thread_id")
            t0 = time.perf_counter()
            logger.info(
                "node_start",
                node=node_name,
                thread_id=thread_id,
                current_status=state.get("status"),
                message_count=len(state.get("messages", [])),
            )
            try:
                result = await fn(state, config)
                duration_ms = round((time.perf_counter() - t0) * 1000)
                logger.info(
                    "node_complete",
                    node=node_name,
                    thread_id=thread_id,
                    duration_ms=duration_ms,
                    new_status=result.get("status") if isinstance(result, dict) else None,
                )
                return result
            except GraphInterrupt:
                # LangGraph interrupt() signal — must propagate for the runtime to suspend the graph.
                raise
            except Exception as exc:
                duration_ms = round((time.perf_counter() - t0) * 1000)
                info: DeadLetterInfo = {
                    "failed_node": node_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                total_dead_lettered = dead_letter_counter.increment(node_name)
                logger.error(
                    "Node raised unhandled exception, routing to dead_letter",
                    node=node_name,
                    thread_id=thread_id,
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    dead_letter_count_total=total_dead_lettered,
                    dead_letter_count_by_node=dead_letter_counter.snapshot(),
                )
                return {"dead_letter": info, "status": "dead_lettered"}

        wrapper.__name__ = fn.__name__
        return wrapper

    return decorator


def after(next_node: str) -> Callable:
    """Return a routing function that checks for dead_letter before routing to next_node.

    Used as the condition in graph.add_conditional_edges() for every node that
    can raise — this replaces add_edge for those nodes.

    Args:
        next_node: The node to route to when dead_letter is not set.
    """

    def _route(state: dict) -> str:
        return "dead_letter" if state.get("dead_letter") else next_node

    _route.__name__ = f"after_{next_node}"
    return _route


async def dead_letter_node(state: "AgentState", config: RunnableConfig) -> dict:
    """Terminal node: structured-log the DeadLetterInfo and return.

    The checkpointer persists the full state automatically — the record is
    replayable via time-travel from this checkpoint.
    """
    _ = config
    logger.error("dead_letter", dead_letter=state.get("dead_letter"))
    return {}
