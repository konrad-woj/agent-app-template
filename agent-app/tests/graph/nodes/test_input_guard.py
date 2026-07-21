"""Tests for input_guard node — three-layer pipeline."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from app.graph.nodes.input_guard import make_input_guard_node
from tests.graph.nodes.conftest import CONFIG, base_state, make_mock_gliguard, make_mock_llm


class TestInputGuardLayerOne:
    """Layer 1: regex sanitisation blocks before GLiGuard or LLM are called."""

    async def test_null_bytes_blocked_before_gliguard(self) -> None:
        gliguard = make_mock_gliguard()
        llm = make_mock_llm('{"verdict": "safe", "reason": "ok"}')
        node = make_input_guard_node(llm, gliguard)

        state = base_state(messages=[HumanMessage(content="\x00\x00\x00")])
        result = await node(state, CONFIG)

        assert result["status"] == "blocked"
        assert "sanitiser" in result["guard_reason"].lower()
        gliguard.acheck_input.assert_not_called()
        llm.ainvoke.assert_not_called()

    async def test_xml_injection_blocked_before_gliguard(self) -> None:
        gliguard = make_mock_gliguard()
        llm = make_mock_llm('{"verdict": "safe", "reason": "ok"}')
        node = make_input_guard_node(llm, gliguard)

        # Only XML tags — sanitiser strips them and the result is empty
        state = base_state(messages=[HumanMessage(content="<system></system>")])
        result = await node(state, CONFIG)

        assert result["status"] == "blocked"
        gliguard.acheck_input.assert_not_called()


class TestInputGuardLayerTwo:
    """Layer 2: GLiGuard injection/jailbreak detection blocks before LLM."""

    async def test_injection_detected_by_gliguard_blocks_before_llm(self) -> None:
        gliguard = make_mock_gliguard(blocked=True, reason="Prompt injection detected.")
        llm = make_mock_llm('{"verdict": "safe", "reason": "ok"}')
        node = make_input_guard_node(llm, gliguard)

        result = await node(base_state(), CONFIG)

        assert result["status"] == "blocked"
        assert "injection" in result["guard_reason"].lower()
        llm.ainvoke.assert_not_called()

    async def test_gliguard_pass_proceeds_to_llm(self) -> None:
        gliguard = make_mock_gliguard(blocked=False)
        llm = make_mock_llm('{"verdict": "safe", "reason": "Valid research question."}')
        node = make_input_guard_node(llm, gliguard)

        result = await node(base_state(), CONFIG)

        assert result["status"] == "planning"
        llm.ainvoke.assert_called_once()


class TestInputGuardLayerThree:
    """Layer 3: LLM topic check blocks off-topic requests that pass layers 1 and 2."""

    async def test_passes_valid_research_request(self) -> None:
        gliguard = make_mock_gliguard()
        llm = make_mock_llm('{"verdict": "safe", "reason": "Valid research question."}')
        node = make_input_guard_node(llm, gliguard)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "planning"
        assert "guard_reason" not in result or result.get("guard_reason") is None

    async def test_blocks_off_topic_request(self) -> None:
        gliguard = make_mock_gliguard()
        llm = make_mock_llm('{"verdict": "unsafe", "reason": "Request is off-topic for research."}')
        node = make_input_guard_node(llm, gliguard)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "blocked"
        assert result["guard_reason"] == "Request is off-topic for research."

    async def test_blocks_on_unparseable_llm_response(self) -> None:
        gliguard = make_mock_gliguard()
        llm = make_mock_llm("I cannot determine the relevance of this request.")
        node = make_input_guard_node(llm, gliguard)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "blocked"
        assert "guard_reason" in result

    async def test_dead_letter_on_llm_error(self) -> None:
        from app.exceptions import LLMServiceUnavailableError

        gliguard = make_mock_gliguard()
        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        node = make_input_guard_node(llm, gliguard)
        result = await node(base_state(), CONFIG)
        assert result["status"] == "dead_lettered"
        assert result["dead_letter"]["failed_node"] == "input_guard"
