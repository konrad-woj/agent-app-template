"""Tests for writer node."""

from unittest.mock import AsyncMock, MagicMock

from app.exceptions import LLMServiceUnavailableError
from app.graph.nodes.writer import make_writer_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_llm


class TestWriterNode:
    async def test_extracts_claims_from_json_response(self) -> None:
        payload = '{"answer": "The internet was invented in 1969.", "claims": ["ARPANET launched in 1969.", "TCP/IP standardised in 1983."]}'
        node = make_writer_node(make_mock_llm(payload))
        state = base_state(plan=["1. Search history", "2. Analyse milestones"])
        result = await node(state, CONFIG)

        assert result["draft_answer"] == "The internet was invented in 1969."
        assert result["claims"] == ["ARPANET launched in 1969.", "TCP/IP standardised in 1983."]
        assert result["status"] == "writing"

    async def test_parse_failure_falls_back_to_raw_and_empty_claims(self) -> None:
        raw = "This is a comprehensive answer to the question."
        node = make_writer_node(make_mock_llm(raw))
        state = base_state(plan=["1. Search for X"])
        result = await node(state, CONFIG)

        assert result["draft_answer"] == raw
        assert result["claims"] == []
        assert result["status"] == "writing"

    async def test_dead_letter_on_llm_error(self) -> None:
        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_writer_node(llm)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "writer"
        assert result["dead_letter"]["error_type"] == "LLMServiceUnavailableError"
