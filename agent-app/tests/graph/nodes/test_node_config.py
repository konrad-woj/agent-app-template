"""Tests for NodeLLMConfig and per-node override behaviour in build_llm / invoke wrappers."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm, llm_invoke, llm_invoke_with_retry

_CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}
_MESSAGES: list[AnyMessage] = [HumanMessage(content="hello")]
_RESPONSE = AIMessage(content="hi")


class TestNodeLLMConfig:
    def test_defaults_are_none(self) -> None:
        cfg = NodeLLMConfig()
        assert cfg.model is None
        assert cfg.temperature is None
        assert cfg.thinking is None
        assert cfg.timeout_seconds is None
        assert cfg.max_retries is None

    def test_fields_set_correctly(self) -> None:
        cfg = NodeLLMConfig(model="openai/gpt-4o", temperature=0.2, thinking=True, timeout_seconds=30.0, max_retries=1)
        assert cfg.model == "openai/gpt-4o"
        assert cfg.temperature == 0.2
        assert cfg.thinking is True
        assert cfg.timeout_seconds == 30.0
        assert cfg.max_retries == 1


class TestBuildLlmOverride:
    def test_no_override_uses_settings(self) -> None:
        with patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls:
            build_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["metadata"]["timeout_seconds"] is not None
        assert kwargs["metadata"]["max_retries"] is not None

    def test_override_timeout_stored_in_metadata(self) -> None:
        with patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls:
            build_llm(NodeLLMConfig(timeout_seconds=15.0))
        _, kwargs = mock_cls.call_args
        assert kwargs["metadata"]["timeout_seconds"] == 15.0

    def test_override_max_retries_stored_in_metadata(self) -> None:
        with patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls:
            build_llm(NodeLLMConfig(max_retries=1))
        _, kwargs = mock_cls.call_args
        assert kwargs["metadata"]["max_retries"] == 1

    def test_override_thinking_passed_to_model_kwargs(self) -> None:
        with patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls:
            build_llm(NodeLLMConfig(thinking=True))
        _, kwargs = mock_cls.call_args
        assert kwargs["model_kwargs"]["enable_thinking"] is True

    def test_override_model_passed(self) -> None:
        with patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls:
            build_llm(NodeLLMConfig(model="openai/gpt-4o-mini"))
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "openai/gpt-4o-mini"

    def test_no_override_model_uses_settings(self) -> None:
        with (
            patch("app.graph.nodes._llm_invoke.ChatLiteLLM") as mock_cls,
            patch("app.graph.nodes._llm_invoke.settings") as mock_settings,
        ):
            mock_settings.llm_model = "openai/default-model"
            mock_settings.llm_base_url = "http://localhost/v1"
            mock_settings.llm_api_key = None
            mock_settings.llm_thinking = False
            mock_settings.llm_timeout_seconds = 60.0
            mock_settings.llm_max_retries = 3
            build_llm()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "openai/default-model"


class TestLlmInvokeMetadataOverride:
    async def test_uses_timeout_from_metadata(self) -> None:
        llm = MagicMock()
        llm.metadata = {"timeout_seconds": 0.001, "max_retries": 3}

        async def _slow(*args, **kwargs):
            import asyncio

            await asyncio.sleep(10)

        llm.ainvoke = _slow
        from app.exceptions import LLMServiceUnavailableError

        with pytest.raises(LLMServiceUnavailableError, match="timed out"):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_uses_max_retries_from_metadata(self) -> None:
        llm = MagicMock()
        from litellm.exceptions import ServiceUnavailableError

        llm.metadata = {"timeout_seconds": 60.0, "max_retries": 1}
        llm.ainvoke = AsyncMock(side_effect=ServiceUnavailableError("503", llm_provider="test", model="test"))

        from app.exceptions import LLMServiceUnavailableError

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(LLMServiceUnavailableError),
        ):
            await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)

        assert llm.ainvoke.await_count == 2  # 1 attempt + 1 retry

    async def test_metadata_overrides_settings_for_retries(self) -> None:
        """Node-level max_retries=0 retries zero times even if settings says 3."""
        llm = MagicMock()
        from litellm.exceptions import RateLimitError

        llm.metadata = {"timeout_seconds": 60.0, "max_retries": 0}
        llm.ainvoke = AsyncMock(side_effect=RateLimitError("429", llm_provider="test", model="test"))

        from app.exceptions import LLMRateLimitError

        with pytest.raises(LLMRateLimitError):
            await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)

        llm.ainvoke.assert_awaited_once()


def _make_null_gliguard() -> MagicMock:
    from app.guards.gliguard import GLiGuardClient, GuardResult

    g = MagicMock(spec=GLiGuardClient)
    g.check_input.return_value = GuardResult(blocked=False)
    g.check_output.return_value = GuardResult(blocked=False)
    g.acheck_input = AsyncMock(return_value=GuardResult(blocked=False))
    g.acheck_output = AsyncMock(return_value=GuardResult(blocked=False))
    return g


class TestCompileGraphNodeLLMConfigs:
    def test_build_llm_called_once_without_configs(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from app.graph.workflow import compile_graph

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)

        with patch("app.graph.workflow.build_llm", return_value=fake_llm) as mock_build_llm:
            compile_graph(InMemorySaver(), mcp_tools=[], gliguard=_make_null_gliguard(), node_llm_configs=None)

        mock_build_llm.assert_called_once_with()

    def test_build_llm_called_per_node_config(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from app.graph.workflow import compile_graph

        thinking_cfg = NodeLLMConfig(thinking=True)
        fast_cfg = NodeLLMConfig(thinking=False, timeout_seconds=10.0)
        node_llm_configs = {
            "planner": thinking_cfg,
            "react_researcher": thinking_cfg,
            "writer": fast_cfg,
        }

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)

        with patch("app.graph.workflow.build_llm", return_value=fake_llm) as mock_build_llm:
            compile_graph(
                InMemorySaver(), mcp_tools=[], gliguard=_make_null_gliguard(), node_llm_configs=node_llm_configs
            )

        # First call: default_llm with no args; then one call per node config entry
        assert mock_build_llm.call_args_list[0] == call()
        config_calls = [c.args[0] for c in mock_build_llm.call_args_list[1:]]
        assert thinking_cfg in config_calls
        assert fast_cfg in config_calls
        assert len(config_calls) == 3

    def test_writer_gets_nonzero_temperature(self) -> None:
        """Writer node should get a nonzero temperature; structured nodes should get 0."""
        from langgraph.checkpoint.memory import InMemorySaver

        from app.graph.workflow import compile_graph

        node_llm_configs: dict[str, NodeLLMConfig] = {
            "input_guard": NodeLLMConfig(temperature=0.0),
            "planner": NodeLLMConfig(temperature=0.0),
            "react_researcher": NodeLLMConfig(temperature=0.0),
            "writer": NodeLLMConfig(temperature=0.3),
            "reflection": NodeLLMConfig(temperature=0.2),
            "output_guard": NodeLLMConfig(temperature=0.0),
        }

        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_llm)

        with patch("app.graph.workflow.build_llm", return_value=fake_llm) as mock_build_llm:
            compile_graph(
                InMemorySaver(), mcp_tools=[], gliguard=_make_null_gliguard(), node_llm_configs=node_llm_configs
            )

        all_calls = mock_build_llm.call_args_list
        # One default call + 6 per-node calls
        assert len(all_calls) == 7
        cfg_by_name = dict(zip(node_llm_configs.keys(), [c.args[0] for c in all_calls[1:]], strict=False))
        assert cfg_by_name["writer"].temperature == 0.3
        assert cfg_by_name["reflection"].temperature == 0.2
        assert cfg_by_name["input_guard"].temperature == 0.0
        assert cfg_by_name["planner"].temperature == 0.0
        assert cfg_by_name["react_researcher"].temperature == 0.0
        assert cfg_by_name["output_guard"].temperature == 0.0
