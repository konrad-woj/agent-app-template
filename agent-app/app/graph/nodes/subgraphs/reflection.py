"""Reflection subgraph: critic/refiner loop with a hard attempt ceiling.

Internal state keys (draft, passed) differ from AgentState (draft_answer,
reflection_passed), so this subgraph must be invoked through a wrapper node
in workflow.py that maps keys in both directions.

Loop exit conditions (whichever comes first):
  1. critic sets passed=True
  2. reflection_attempts reaches MAX_REFLECTION_ATTEMPTS (settings.max_reflection_attempts)

When the ceiling is hit, the best draft available is kept and reflection_passed
propagates as False — the output guard still runs.

Factory functions accept an LLM so tests can inject a mock.
"""

import time
from collections.abc import Callable
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from logger import get_logger
from pydantic import BaseModel, field_validator

from app.config import settings
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.prompts.loader import load_system, render_human

logger = get_logger(__name__)

_CRITIC_PROMPT = load_system("reflection", "critic")
_REFINER_PROMPT = load_system("reflection", "refiner")


class ReflectionState(TypedDict):
    draft: str
    critique: str
    reflection_attempts: int
    passed: bool


class _CriticResponse(BaseModel):
    verdict: str
    critique: str

    @field_validator("verdict")
    @classmethod
    def normalise_verdict(cls, v: str) -> str:
        return v.strip().lower()


def make_critic_node(llm: BaseChatModel) -> Callable:
    """Return an async critic node bound to the given LLM."""

    async def critic_node(state: ReflectionState, config: RunnableConfig) -> dict:
        attempt = state["reflection_attempts"] + 1
        logger.info("critic.start", attempt=attempt, draft_length=len(state["draft"]))
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_CRITIC_PROMPT),
            HumanMessage(content=render_human("reflection", "critic", draft=state["draft"])),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        parsed = parse_structured(str(response.content), _CriticResponse)
        if parsed is not None:
            passed = parsed.verdict == "pass"
            critique = parsed.critique
        else:
            passed = True  # parse failure → treat as pass to avoid infinite loop
            critique = "Could not parse critic response."
        duration_ms = round((time.perf_counter() - t0) * 1000)
        logger.info("critic.complete", attempt=attempt, passed=passed, critique=critique, duration_ms=duration_ms)
        return {
            "critique": critique,
            "passed": passed,
            "reflection_attempts": attempt,
        }

    return critic_node


def make_refiner_node(llm: BaseChatModel) -> Callable:
    """Return an async refiner node bound to the given LLM."""

    async def refiner_node(state: ReflectionState, config: RunnableConfig) -> dict:
        logger.info(
            "refiner.start",
            attempt=state["reflection_attempts"],
            draft_length=len(state["draft"]),
            critique_length=len(state["critique"]),
        )
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_REFINER_PROMPT),
            HumanMessage(
                content=render_human("reflection", "refiner", draft=state["draft"], critique=state["critique"])
            ),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        refined = str(response.content)
        duration_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "refiner.complete", attempt=state["reflection_attempts"], draft_length=len(refined), duration_ms=duration_ms
        )
        return {"draft": refined}

    return refiner_node


def should_refine(state: ReflectionState) -> Literal["refiner", "__end__"]:
    ceiling_hit = state["reflection_attempts"] >= settings.max_reflection_attempts
    return "__end__" if state["passed"] or ceiling_hit else "refiner"


def build_reflection_subgraph(llm: BaseChatModel) -> CompiledStateGraph:
    """Build and compile the reflection subgraph with the given LLM.

    Args:
        llm: LLM used for critic and refiner nodes.
    """
    graph: StateGraph = StateGraph(ReflectionState)
    graph.add_node("critic", make_critic_node(llm))
    graph.add_node("refiner", make_refiner_node(llm))
    graph.add_edge(START, "critic")
    graph.add_conditional_edges("critic", should_refine)
    graph.add_edge("refiner", "critic")
    return graph.compile()
