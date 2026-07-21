"""Verify subgraph: fan-out/fan-in claim verification using the Send API.

Demonstrates:
  - Fan-out: ``route_to_verifiers`` returns one ``Send("verifier", ...)`` per claim.
    LangGraph spawns each branch in parallel.
  - Fan-in: ``VerificationState.results`` carries ``Annotated[list[dict], operator.add]``.
    LangGraph calls operator.add to merge results from all parallel branches before
    the subgraph exits.

Each branch is genuinely multi-step, which justifies the Send API over ToolNode:
  1. ``fact_check`` tool call — web search + top-source fetch for the claim.
  2. LLM verification — reads evidence, emits a structured verdict.

ToolNode cannot model in-branch LLM reasoning; Send API branches can.

The subgraph is invoked from the parent graph via a wrapper node
(``_make_run_verification``) in workflow.py that maps AgentState ↔ VerificationState.
"""

import asyncio
import operator
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from logger import get_logger
from pydantic import BaseModel, field_validator

from app.config import settings
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.prompts.loader import load_system, render_human

logger = get_logger(__name__)

_VERIFY_PROMPT = load_system("verification", "verify")


class VerificationState(TypedDict):
    claims: list[str]  # fan-out source: set of claims from writer
    claim: str  # per-branch: set by Send, consumed by verifier_node
    results: Annotated[list[dict], operator.add]  # fan-in target


class _VerifyResult(BaseModel):
    supported: bool
    confidence: str
    reason: str

    @field_validator("confidence")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().lower()


def make_verifier_node(llm: BaseChatModel, fact_check_tool: BaseTool | None) -> Callable:
    """Return an async verifier node bound to the given LLM and optional fact_check tool."""

    async def verifier_node(state: VerificationState, config: RunnableConfig) -> dict:
        claim = state["claim"]
        logger.info("verifier.start", claim_length=len(claim))

        # Step 1: tool call — gather evidence. Bounded by a timeout like every LLM
        # call in this codebase; a hung or failing MCP call raises here instead of
        # blocking the branch indefinitely, so the wrapper node's dead-letter
        # handling in workflow.py can catch it.
        if fact_check_tool is not None:
            evidence = await asyncio.wait_for(
                fact_check_tool.ainvoke({"claim": claim}),
                timeout=settings.mcp_tool_call_timeout_seconds,
            )
        else:
            evidence = "Verification skipped — no fact_check tool available."

        # Step 2: LLM verification — structured verdict from evidence.
        messages = [
            SystemMessage(content=_VERIFY_PROMPT),
            HumanMessage(content=render_human("verification", "verify", claim=claim, evidence=evidence)),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        parsed = parse_structured(str(response.content), _VerifyResult)
        if parsed is not None:
            result = {
                "claim": claim,
                "supported": parsed.supported,
                "confidence": parsed.confidence,
                "reason": parsed.reason,
            }
        else:
            # Fail-open: parse error → treat as supported to avoid false blocks.
            logger.warning("verifier.parse_failed", claim_length=len(claim))
            result = {
                "claim": claim,
                "supported": True,
                "confidence": "low",
                "reason": "Could not parse verification response.",
            }

        logger.info("verifier.complete", supported=result["supported"], confidence=result["confidence"])
        return {"results": [result]}

    return verifier_node


def route_to_verifiers(state: VerificationState) -> list[Send]:
    """Fan-out: spawn one verifier branch per claim."""
    return [Send("verifier", {"claims": [], "claim": c, "results": []}) for c in state["claims"]]


def build_verify_subgraph(llm: BaseChatModel, fact_check_tool: BaseTool | None) -> CompiledStateGraph:
    """Build and compile the verification subgraph with the given LLM and tool.

    Args:
        llm: LLM used for structured verdict in each verifier branch.
        fact_check_tool: MCP fact_check tool; if None, branches skip the tool call.
    """
    graph: StateGraph = StateGraph(VerificationState)
    graph.add_node("router", lambda state: state)
    graph.add_node("verifier", make_verifier_node(llm, fact_check_tool))
    graph.add_conditional_edges("router", route_to_verifiers)
    graph.add_edge(START, "router")
    graph.add_edge("verifier", END)
    return graph.compile()
