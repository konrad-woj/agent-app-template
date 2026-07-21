"""Planner + plan_review nodes: generate a research plan, then pause for human approval.

Split into two nodes because LangGraph re-executes a node's entire function body
from the top on every resume — only the *return value* is checkpointed, not
progress within the function. A single node that generates the plan and then
calls interrupt() would regenerate the plan (a fresh, nondeterministic LLM call)
every time the graph resumes, so the plan the user approved in the interrupt
payload could differ from the plan actually used afterward. Splitting avoids
that: all nondeterministic work happens in `planner`, which returns before any
interrupt; `plan_review` does nothing but read the already-computed plan from
state and call interrupt() — safe to re-run on resume since it has no side effects.

Demonstrates the LangGraph interrupt() pattern:
  1. planner: LLM generates a plan, then GLiGuard (injection/content check) + LLM
     quality check guard it. If unsafe: returns status="blocked" WITHOUT calling
     interrupt() — the unsafe plan is never surfaced to the user.
  2. plan_review: if the plan was safe, calls interrupt({"plan": plan}) — graph
     suspends here, checkpointed. Resume guard and search run on resumption.
  3. Caller resumes with Command(resume=True) → plan_approved=True → routes to resume_guard.
  4. Caller resumes with Command(resume=False) → plan_approved=False → routes to END (aborted).

Both nodes are wrapped with with_dead_letter so any error routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from logger import get_logger

from app.config import settings
from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.graph.nodes._messages import get_last_human_text
from app.guards.gliguard import GLiGuardClient
from app.prompts.loader import load_system, render_human

logger = get_logger(__name__)

_PLAN_SYSTEM_PROMPT = load_system("planner", "plan")
_PLAN_GUARD_PROMPT = load_system("planner", "plan_guard")


def make_planner_node(llm: BaseChatModel, gliguard: GLiGuardClient) -> Callable:
    """Return an async planner node bound to the given LLM and GLiGuardClient."""

    @with_dead_letter("planner")
    async def planner(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        user_text = get_last_human_text(state["messages"])
        logger.info("planner.inputs", question_length=len(user_text), message_count=len(state["messages"]))

        messages = [
            SystemMessage(content=_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=str(user_text)),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        plan_text = str(response.content)
        plan_steps = [line.strip() for line in plan_text.splitlines() if line.strip()]
        logger.info("planner.plan_generated", step_count=len(plan_steps))

        # Guard the generated plan before surfacing it to the user.
        plan_as_text = "\n".join(plan_steps)

        # GLiGuard injection check on plan text.
        guard_result = await gliguard.acheck_input(plan_as_text, settings.guard_timeout_seconds)
        if guard_result.blocked:
            logger.info("planner.guard_gliguard_blocked", reason=guard_result.reason)
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": guard_result.reason or "Generated plan failed safety check.",
            }

        # LLM quality + safety check on plan text.
        guard_messages = [
            SystemMessage(content=_PLAN_GUARD_PROMPT),
            HumanMessage(content=render_human("planner", "plan_guard", plan=plan_as_text)),
        ]
        guard_response = await llm_invoke_with_retry(llm, guard_messages, config)
        verdict = parse_structured(str(guard_response.content), GuardVerdict)
        if verdict is None:
            logger.warning("planner.guard_llm_parse_failed")
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": "Plan guard could not parse LLM response — treating as unsafe.",
            }

        if verdict.verdict != "safe":
            logger.info("planner.guard_llm_blocked", reason=verdict.reason)
            return {
                "plan": plan_steps,
                "status": "blocked",
                "guard_reason": verdict.reason,
            }

        logger.info("planner.guard_passed", awaiting_approval=True)
        return {"plan": plan_steps, "status": "awaiting_approval"}

    return planner


def make_plan_review_node() -> Callable:
    """Return an async node that pauses for human approval of an already-guarded plan.

    Reads ``plan`` from state (written by ``planner``) and does nothing else
    before calling interrupt() — safe to re-run from the top on resume.
    """

    @with_dead_letter("plan_review")
    async def plan_review(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        _ = config
        plan_steps = state.get("plan", [])
        approved: bool = interrupt({"plan": plan_steps})

        logger.info("plan_review.resumed", approved=approved)
        if not approved:
            return {"plan_approved": False, "status": "aborted"}
        return {"plan_approved": True, "status": "researching"}

    return plan_review
