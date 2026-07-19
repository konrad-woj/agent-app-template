"""Tests for verify_subgraph: fan-out/fan-in claim verification."""

from unittest.mock import AsyncMock, MagicMock

from app.graph.nodes.subgraphs.verification import (
    VerificationState,
    build_verify_subgraph,
    make_verifier_node,
)
from tests.graph.nodes.conftest import CONFIG, make_mock_llm

_SUPPORTED_JSON = '{"supported": true, "confidence": "high", "reason": "Evidence clearly supports the claim."}'
_UNSUPPORTED_JSON = '{"supported": false, "confidence": "high", "reason": "No evidence found."}'


class TestVerifierNode:
    async def test_supported_claim(self) -> None:
        tool = AsyncMock(return_value="Evidence text from web search.")
        fact_check_tool = MagicMock()
        fact_check_tool.ainvoke = tool

        node = make_verifier_node(make_mock_llm(_SUPPORTED_JSON), fact_check_tool)
        state: VerificationState = {"claims": [], "claim": "ARPANET launched in 1969.", "results": []}
        result = await node(state, CONFIG)

        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["supported"] is True
        assert r["confidence"] == "high"
        assert r["claim"] == "ARPANET launched in 1969."

    async def test_unsupported_claim(self) -> None:
        tool = AsyncMock(return_value="No relevant evidence found.")
        fact_check_tool = MagicMock()
        fact_check_tool.ainvoke = tool

        node = make_verifier_node(make_mock_llm(_UNSUPPORTED_JSON), fact_check_tool)
        state: VerificationState = {"claims": [], "claim": "Claim that is false.", "results": []}
        result = await node(state, CONFIG)

        r = result["results"][0]
        assert r["supported"] is False
        assert "No evidence found." in r["reason"]

    async def test_parse_failure_fails_open(self) -> None:
        """Garbage LLM response → supported=True, confidence=low (fail-open)."""
        node = make_verifier_node(make_mock_llm("This looks correct to me."), None)
        state: VerificationState = {"claims": [], "claim": "Some claim.", "results": []}
        result = await node(state, CONFIG)

        r = result["results"][0]
        assert r["supported"] is True
        assert r["confidence"] == "low"

    async def test_no_tool_skips_call(self) -> None:
        """fact_check_tool=None → node still runs, returns a result."""
        node = make_verifier_node(make_mock_llm(_SUPPORTED_JSON), None)
        state: VerificationState = {"claims": [], "claim": "Some claim.", "results": []}
        result = await node(state, CONFIG)
        assert len(result["results"]) == 1

    async def test_confidence_normalised_to_lowercase(self) -> None:
        payload = '{"supported": true, "confidence": "HIGH", "reason": "ok"}'
        node = make_verifier_node(make_mock_llm(payload), None)
        state: VerificationState = {"claims": [], "claim": "claim", "results": []}
        result = await node(state, CONFIG)
        assert result["results"][0]["confidence"] == "high"


class TestVerifySubgraph:
    async def test_fanout_n_claims_produces_n_results(self) -> None:
        llm = make_mock_llm(_SUPPORTED_JSON)
        subgraph = build_verify_subgraph(llm, None)
        input_state: VerificationState = {
            "claims": ["claim A", "claim B", "claim C"],
            "claim": "",
            "results": [],
        }
        result = await subgraph.ainvoke(input_state, CONFIG)
        assert len(result["results"]) == 3

    async def test_empty_claims_produces_empty_results(self) -> None:
        llm = make_mock_llm(_SUPPORTED_JSON)
        subgraph = build_verify_subgraph(llm, None)
        input_state: VerificationState = {"claims": [], "claim": "", "results": []}
        result = await subgraph.ainvoke(input_state, CONFIG)
        assert result["results"] == []

    async def test_each_result_contains_claim_text(self) -> None:
        llm = make_mock_llm(_SUPPORTED_JSON)
        subgraph = build_verify_subgraph(llm, None)
        claims = ["fact one", "fact two"]
        input_state: VerificationState = {"claims": claims, "claim": "", "results": []}
        result = await subgraph.ainvoke(input_state, CONFIG)
        returned_claims = {r["claim"] for r in result["results"]}
        assert returned_claims == set(claims)
