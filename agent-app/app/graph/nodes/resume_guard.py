"""Resume guard node: validates the human's resume message before search begins.

Positioned between planner (approved path) and search_subgraph. Applies layers 1
and 2 of the guard pipeline to the message the user sent alongside approve=True/False.
Layer 3 (LLM topic check) is omitted — topic was already validated by input_guard on
the first turn.

The resume message is written to state["messages"] by the router via aupdate_state
before Command(resume=...) is sent, so this node reads state["messages"][-1].

On safe:   routes to react_researcher via _resume_guard_condition in workflow.py.
On block:  sets status="blocked", guard_reason=<reason>, routes to END.
On error:  with_dead_letter catches the exception and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from logger import get_logger

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_layers import run_sanitize_and_injection_check
from app.graph.nodes._messages import get_last_human_text
from app.guards.gliguard import GLiGuardClient

logger = get_logger(__name__)


def make_resume_guard_node(gliguard: GLiGuardClient) -> Callable:
    """Return an async resume_guard node bound to the given GLiGuardClient."""

    @with_dead_letter("resume_guard")
    async def resume_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        raw_text = get_last_human_text(state["messages"])
        logger.info("resume_guard.inputs", input_text_length=len(raw_text))

        # Layers 1-2: regex sanitisation, then GLiGuard injection/jailbreak check.
        result = await run_sanitize_and_injection_check(
            gliguard, raw_text, "resume_guard", "Resume message", "Resume message failed injection check."
        )
        if isinstance(result, dict):
            return result

        return {}

    return resume_guard
