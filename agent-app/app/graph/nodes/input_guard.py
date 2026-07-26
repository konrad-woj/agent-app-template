"""Input guardrail node: three-layer safety and topic check.

Runs first in the graph (START → input_guard). Three layers applied in order,
short-circuiting on block:

  1. Regex blocklist (_prompt_utils.sanitize_user_text) — null bytes, XML injection
     markers, tool-call syntax; <1ms, zero model cost.
  2. GLiGuard (fastino/gliguard-LLMGuardrails-300M) — prompt injection and jailbreak;
     ~15ms GPU; no topic knowledge.
  3. LLM topic check — research-domain relevance only (safety is owned by layer 2);
     ~300ms.

Layer 1 only ever strips content out of the raw text — it never adds anything
back in — so once sanitisation changes the text, the original HumanMessage in
state is replaced with the cleaned version via RemoveMessage(id=...). Without
this, downstream nodes (planner, writer) would re-read the raw, unsanitised
text straight out of state via get_last_human_text, silently undoing layer 1.

On safe:   routes to planner via after("planner") in workflow.py.
On block:  sets status="blocked", guard_reason=<reason>, routes to END.
On error:  with_dead_letter catches the exception and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from logger import get_logger

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._guard_layers import run_sanitize_and_injection_check
from app.graph.nodes._guard_verdict import GuardVerdict
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.graph.nodes._messages import get_last_human_message
from app.guards.gliguard import GLiGuardClient
from app.prompts.loader import load_system

logger = get_logger(__name__)

_TOPIC_CHECK_PROMPT = load_system("input_guard", "topic_check")


def make_input_guard_node(llm: BaseChatModel, gliguard: GLiGuardClient) -> Callable:
    """Return an async input_guard node bound to the given LLM and GLiGuardClient."""

    @with_dead_letter("input_guard")
    async def input_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        last_human = get_last_human_message(state["messages"])
        raw_text = str(last_human.content) if last_human else ""
        logger.info("input_guard.inputs", input_text_length=len(raw_text))

        # Layers 1-2: regex sanitisation, then GLiGuard injection/jailbreak detection.
        result = await run_sanitize_and_injection_check(
            gliguard, raw_text, "input_guard", "Input", "Potential prompt injection detected."
        )
        if isinstance(result, dict):
            return result
        user_text = result

        sanitized_messages: list[AnyMessage | RemoveMessage] = []
        if last_human is not None and user_text != raw_text:
            assert last_human.id is not None
            sanitized_messages = [RemoveMessage(id=last_human.id), HumanMessage(content=user_text)]

        # Layer 3: LLM topic relevance check (not safety — GLiGuard owns that).
        messages = [
            SystemMessage(content=_TOPIC_CHECK_PROMPT),
            HumanMessage(content=user_text),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        verdict = parse_structured(str(response.content), GuardVerdict)
        if verdict is None:
            logger.warning("input_guard.layer3_parse_failed")
            return {
                "status": "blocked",
                "guard_reason": "Input guard could not parse LLM response — treating as off-topic.",
                "messages": sanitized_messages,
            }

        logger.info("input_guard.layer3_verdict", verdict=verdict.verdict, reason=verdict.reason)
        if verdict.verdict == "safe":
            return {"status": "planning", "messages": sanitized_messages}
        return {"status": "blocked", "guard_reason": verdict.reason, "messages": sanitized_messages}

    return input_guard
