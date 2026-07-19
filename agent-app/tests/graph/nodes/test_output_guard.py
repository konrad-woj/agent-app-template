"""Tests for output_guard node: PII redaction + deterministic verification check."""

from unittest.mock import AsyncMock, MagicMock

from app.graph.nodes.output_guard import make_output_guard_node
from app.guards.gliguard import GuardResult, Span
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_gliguard


class TestOutputGuardPIIRedaction:
    """Layer 1: GLiGuard PII detection and redaction."""

    async def test_email_in_answer_is_redacted(self) -> None:
        email_span = Span(text="user@example.com", entity_type="email", start=12, end=28)
        gliguard = MagicMock()
        gliguard.acheck_output = AsyncMock(return_value=GuardResult(blocked=False, flagged_spans=[email_span]))
        node = make_output_guard_node(gliguard)

        state = base_state(
            final_answer="Contact us: user@example.com for more.",
            verification_results=[{"claim": "x", "supported": True, "confidence": "high", "reason": "ok"}],
        )
        result = await node(state, CONFIG)

        assert result["status"] == "done"
        assert "[REDACTED:email]" in result["final_answer"]
        assert "user@example.com" not in result["final_answer"]

    async def test_no_pii_answer_unchanged(self) -> None:
        gliguard = make_mock_gliguard()
        node = make_output_guard_node(gliguard)

        state = base_state(final_answer="Clean research answer.")
        result = await node(state, CONFIG)

        assert result["status"] == "done"
        assert result["final_answer"] == "Clean research answer."

    async def test_pii_redacted_before_verification_check(self) -> None:
        """Redacted answer is what gets returned and passed through verification."""
        phone_span = Span(text="555-1234", entity_type="phone number", start=6, end=14)
        gliguard = MagicMock()
        gliguard.acheck_output = AsyncMock(return_value=GuardResult(blocked=False, flagged_spans=[phone_span]))
        node = make_output_guard_node(gliguard)

        state = base_state(final_answer="Call: 555-1234 for info.")
        result = await node(state, CONFIG)

        assert "555-1234" not in result["final_answer"]
        assert "[REDACTED:phone number]" in result["final_answer"]


class TestOutputGuardVerificationCheck:
    """Layer 2: deterministic check against verify_subgraph results."""

    async def test_all_claims_supported_passes(self) -> None:
        gliguard = make_mock_gliguard()
        node = make_output_guard_node(gliguard)
        state = base_state(
            final_answer="Clean research answer.",
            verification_results=[
                {"claim": "A", "supported": True, "confidence": "high", "reason": "Evidence found."},
                {"claim": "B", "supported": True, "confidence": "medium", "reason": "Partially supported."},
            ],
        )
        result = await node(state, CONFIG)
        assert result["status"] == "done"
        assert result["final_answer"] == "Clean research answer."

    async def test_unsupported_claim_blocks(self) -> None:
        gliguard = make_mock_gliguard()
        node = make_output_guard_node(gliguard)
        state = base_state(
            final_answer="Answer with a bad claim.",
            verification_results=[
                {"claim": "A", "supported": True, "confidence": "high", "reason": "ok"},
                {"claim": "B", "supported": False, "confidence": "high", "reason": "No evidence found."},
            ],
        )
        result = await node(state, CONFIG)
        assert result["status"] == "blocked"
        assert "unable to produce" in result["final_answer"].lower()
        assert "No evidence found." in result["guard_reason"]

    async def test_empty_verification_results_passes(self) -> None:
        """Writer parse failure leaves claims=[] → verify_subgraph fans to zero → still passes."""
        gliguard = make_mock_gliguard()
        node = make_output_guard_node(gliguard)
        state = base_state(final_answer="Some answer.", verification_results=[])
        result = await node(state, CONFIG)
        assert result["status"] == "done"

    async def test_multiple_unsupported_all_reasons_included(self) -> None:
        gliguard = make_mock_gliguard()
        node = make_output_guard_node(gliguard)
        state = base_state(
            final_answer="Bad answer.",
            verification_results=[
                {"claim": "A", "supported": False, "confidence": "high", "reason": "Reason A."},
                {"claim": "B", "supported": False, "confidence": "low", "reason": "Reason B."},
            ],
        )
        result = await node(state, CONFIG)
        assert result["status"] == "blocked"
        assert "Reason A." in result["guard_reason"]
        assert "Reason B." in result["guard_reason"]
