"""Output guardrail node: PII redaction then deterministic claim verification check.

Runs as the last node before END. Two layers applied in order:

  1. GLiGuard PII detection — detects email, phone, SSN, credit card, API key, IP
     spans in final_answer; redacts in-place with [REDACTED:<entity_type>]. Does NOT
     block — redacts and continues.
  2. Verification check — reads verification_results written by verify_subgraph.
     Blocks if any claim is marked unsupported. No LLM call — deterministic.

On safe:    sets status="done", preserves (redacted) final_answer.
On blocked: sets status="blocked", replaces final_answer with safe fallback message,
            sets guard_reason — does NOT loop back (keeps graph acyclic).
On error:   with_dead_letter catches and routes to dead_letter.
"""

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig
from logger import get_logger

from app.config import settings
from app.graph.nodes._dead_letter import with_dead_letter
from app.guards.gliguard import GLiGuardClient, redact
from app.prompts.loader import load_text

logger = get_logger(__name__)

_SAFE_FALLBACK = load_text("output_guard", "safe_fallback")


def make_output_guard_node(gliguard: GLiGuardClient) -> Callable:
    """Return an async output_guard node bound to the given GLiGuardClient."""

    @with_dead_letter("output_guard")
    async def output_guard(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        _ = config
        answer = state.get("final_answer", state.get("draft_answer", ""))
        verification_results: list[dict] = state.get("verification_results", [])  # type: ignore[assignment]
        logger.info(
            "output_guard.inputs",
            answer_length=len(answer),
            verification_count=len(verification_results),
        )

        # Layer 1: GLiGuard PII detection and in-place redaction.
        pii_result = await gliguard.acheck_output(answer, settings.guard_timeout_seconds)
        if pii_result.flagged_spans:
            logger.info("output_guard.pii_redacted", span_count=len(pii_result.flagged_spans))
            answer = redact(answer, pii_result.flagged_spans)
        else:
            logger.info("output_guard.pii_clean")

        # Layer 2: Deterministic check against verify_subgraph results.
        unsupported = [r for r in verification_results if not r.get("supported", True)]
        if unsupported:
            reasons = "; ".join(r.get("reason", "") for r in unsupported)
            logger.info("output_guard.verification_failed", unsupported_count=len(unsupported))
            return {
                "status": "blocked",
                "final_answer": _SAFE_FALLBACK,
                "guard_reason": f"Answer contains unverified claims: {reasons}",
            }

        logger.info("output_guard.verification_passed", total_claims=len(verification_results))
        return {"status": "done", "final_answer": answer}

    return output_guard
