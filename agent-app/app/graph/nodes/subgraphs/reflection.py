"""Reflection subgraph: critic/refiner loop with a hard attempt ceiling.

Internal state keys (draft, passed) differ from AgentState (draft_answer,
reflection_passed), so this subgraph must be invoked through a wrapper node
in workflow.py that maps keys in both directions.

Loop exit conditions (whichever comes first):
  1. critic's scores clear the pass policy (passed=True)
  2. reflection_attempts reaches MAX_REFLECTION_ATTEMPTS (settings.max_reflection_attempts)
  3. the critic's response can't be parsed (critic_failed=True, passed=False)

When the ceiling is hit, the best draft available is kept and reflection_passed
propagates as False — the output guard still runs.

The critic scores the draft per criterion (reasoning → score → confidence, see
_rubric_models.py) and pass/fail is decided here in code from those scores — the
LLM is never asked for the verdict or the total. Criteria the critic flags as
low-confidence are returned in `low_confidence` so they surface for human
review instead of disappearing into the pass/fail decision.

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
from pydantic import BaseModel

from app.config import settings
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.graph.nodes.subgraphs._rubric_models import (
    CriterionVerdict,
    HolisticCriterionVerdict,
    criterion_verdicts,
    low_confidence_criteria,
    max_score_of,
)
from app.prompts.loader import load_system, render_human

logger = get_logger(__name__)

_CRITIC_PROMPT = load_system("reflection", "critic")
_REFINER_PROMPT = load_system("reflection", "refiner")


class ReflectionState(TypedDict):
    question: str
    draft: str
    critique: str
    reflection_attempts: int
    passed: bool
    low_confidence: list[str]  # criterion ids the latest critic pass was unsure about
    critic_failed: bool  # latest critic response was unparseable → stop, unreviewed


class CriticVerdict(BaseModel):
    """Critic output. 0–4 for criteria you can point at sentences for, 0–2 for holistic ones."""

    relevance: CriterionVerdict
    completeness: CriterionVerdict
    accuracy: HolisticCriterionVerdict  # the critic has no sources; verify_subgraph checks facts
    clarity: HolisticCriterionVerdict
    critique: str


# Pass policy: every criterion at or above its minimum AND the total at or above
# _MIN_TOTAL (out of 4+4+2+2 = 12). Mirrors the ~3/4 bar in evals/configs/scoring_rubric.yaml.
_MIN_SCORES = {"relevance": 3, "completeness": 3, "accuracy": 1, "clarity": 1}
_MIN_TOTAL = 9


def _apply_pass_policy(verdict: CriticVerdict) -> tuple[bool, str]:
    """Return (passed, critique). The critique lists each criterion below its minimum."""
    scores = criterion_verdicts(verdict)
    total = sum(v.score for v in scores.values())
    below = {name: v for name, v in scores.items() if v.score < _MIN_SCORES[name]}
    passed = not below and total >= _MIN_TOTAL
    critique = verdict.critique
    if below:
        details = "\n".join(f"- {name} ({v.score}/{max_score_of(type(v))}): {v.reason}" for name, v in below.items())
        critique = f"{critique}\n{details}"
    return passed, critique


def make_critic_node(llm: BaseChatModel) -> Callable:
    """Return an async critic node bound to the given LLM."""

    async def critic_node(state: ReflectionState, config: RunnableConfig) -> dict:
        attempt = state["reflection_attempts"] + 1
        logger.info("critic.start", attempt=attempt, draft_length=len(state["draft"]))
        t0 = time.perf_counter()
        messages = [
            SystemMessage(content=_CRITIC_PROMPT),
            HumanMessage(
                content=render_human("reflection", "critic", question=state["question"], draft=state["draft"])
            ),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        raw = str(response.content)
        parsed = parse_structured(raw, CriticVerdict)
        if parsed is not None:
            passed, critique = _apply_pass_policy(parsed)
            low_confidence = low_confidence_criteria(parsed)
            scores = {name: v.score for name, v in criterion_verdicts(parsed).items()}
        else:
            # Parse failure → fail closed: the draft is not marked as passed (a question
            # that talks the critic out of its JSON must not switch the gate off), and the
            # loop stops rather than refining against a critique that doesn't exist.
            logger.warning("critic.parse_failed", attempt=attempt, raw_length=len(raw))
            passed, critique, low_confidence, scores = False, "Could not parse critic response.", [], {}
        duration_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "critic.complete",
            attempt=attempt,
            passed=passed,
            scores=scores,
            low_confidence=low_confidence,
            critique=critique,
            duration_ms=duration_ms,
        )
        return {
            "critique": critique,
            "passed": passed,
            "reflection_attempts": attempt,
            "low_confidence": low_confidence,
            "critic_failed": parsed is None,
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
    return "__end__" if state["passed"] or ceiling_hit or state.get("critic_failed") else "refiner"


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
