"""Tests for reflection subgraph: critic/refiner loop."""

from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.graph.nodes.subgraphs.reflection import ReflectionState, build_reflection_subgraph

_CONFIG: RunnableConfig = {"configurable": {"thread_id": "reflection-test"}}


def _make_llm_sequence(*responses: str) -> MagicMock:
    """Return a mock LLM that yields responses in sequence."""
    ai_responses = [AIMessage(content=r) for r in responses]
    llm = MagicMock()
    llm.metadata = None  # ensures (llm.metadata or {}) falls back to settings in llm_invoke
    llm.ainvoke = AsyncMock(side_effect=ai_responses)
    return llm


def _base_state(draft: str = "draft answer", attempts: int = 0) -> ReflectionState:
    return {"draft": draft, "critique": "", "reflection_attempts": attempts, "passed": False}


class TestReflectionSubgraph:
    async def test_passes_on_first_attempt(self) -> None:
        llm = _make_llm_sequence('{"verdict": "pass", "critique": "Excellent answer."}')
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["passed"] is True
        assert result["reflection_attempts"] == 1

    async def test_refines_on_fail_then_passes(self) -> None:
        llm = _make_llm_sequence(
            '{"verdict": "fail", "critique": "Too vague."}',  # critic 1 → fail
            "Improved answer addressing vagueness.",  # refiner
            '{"verdict": "pass", "critique": "Now specific."}',  # critic 2 → pass
        )
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["passed"] is True
        assert result["reflection_attempts"] == 2
        assert "Improved" in result["draft"]

    async def test_exits_at_ceiling_when_never_passes(self) -> None:
        from unittest.mock import patch

        # Always fail
        fail_critic = '{"verdict": "fail", "critique": "Still bad."}'
        refine_response = "refined draft"
        # Enough responses for ceiling iterations: critic × N + refiner × (N-1)
        responses = []
        for _ in range(3):
            responses.append(fail_critic)
            responses.append(refine_response)
        responses.append(fail_critic)  # last critic

        llm = _make_llm_sequence(*responses)
        with patch("app.graph.nodes.subgraphs.reflection.settings") as mock_settings:
            mock_settings.max_reflection_attempts = 3
            subgraph = build_reflection_subgraph(llm)
            result = await subgraph.ainvoke(_base_state(), _CONFIG)

        assert result["passed"] is False
        assert result["reflection_attempts"] >= 3

    async def test_unparseable_critic_response_treated_as_pass(self) -> None:
        llm = _make_llm_sequence("Looks fine to me, good quality!")
        subgraph = build_reflection_subgraph(llm)
        result = await subgraph.ainvoke(_base_state(), _CONFIG)
        assert result["passed"] is True
