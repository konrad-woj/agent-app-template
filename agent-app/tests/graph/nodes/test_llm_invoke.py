"""Tests for _llm_invoke.py: timeout, exception translation, and retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from litellm.exceptions import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError

from app.exceptions import LLMRateLimitError, LLMServiceError, LLMServiceUnavailableError
from app.graph.nodes._llm_invoke import llm_invoke, llm_invoke_with_retry

_CONFIG: RunnableConfig = {"configurable": {"thread_id": "test"}}
_MESSAGES: list[AnyMessage] = [HumanMessage(content="hello")]
_RESPONSE = AIMessage(content="hi")


def _make_llm(side_effect: Exception | None = None, return_value: AIMessage = _RESPONSE) -> MagicMock:
    llm = MagicMock()
    llm.metadata = None  # ensures llm.metadata or {} falls back to settings
    if side_effect:
        llm.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        llm.ainvoke = AsyncMock(return_value=return_value)
    return llm


class TestLlmInvoke:
    async def test_returns_message_on_success(self) -> None:
        llm = _make_llm()
        result = await llm_invoke(llm, _MESSAGES, _CONFIG)
        assert result == _RESPONSE

    async def test_rate_limit_raises_llm_rate_limit_error(self) -> None:
        llm = _make_llm(side_effect=RateLimitError("429", llm_provider="test", model="test"))
        with pytest.raises(LLMRateLimitError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_service_unavailable_raises_llm_service_unavailable(self) -> None:
        llm = _make_llm(side_effect=ServiceUnavailableError("503", llm_provider="test", model="test"))
        with pytest.raises(LLMServiceUnavailableError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_api_connection_error_raises_llm_service_unavailable(self) -> None:
        llm = _make_llm(side_effect=APIConnectionError("conn error", llm_provider="test", model="test"))
        with pytest.raises(LLMServiceUnavailableError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_api_error_raises_llm_service_error(self) -> None:
        llm = _make_llm(side_effect=APIError(500, "internal error", llm_provider="test", model="test"))
        with pytest.raises(LLMServiceError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_httpx_connect_error_raises_llm_service_unavailable(self) -> None:
        llm = _make_llm(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(LLMServiceUnavailableError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_timeout_raises_llm_service_unavailable(self) -> None:
        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(side_effect=TimeoutError("timed out"))
        with pytest.raises(LLMServiceUnavailableError):
            await llm_invoke(llm, _MESSAGES, _CONFIG)

    async def test_asyncio_timeout_raises_llm_service_unavailable(self) -> None:
        import asyncio

        llm = MagicMock()
        llm.metadata = None

        async def _slow(*args, **kwargs):
            await asyncio.sleep(10)

        llm.ainvoke = _slow

        with patch("app.graph.nodes._llm_invoke.settings") as mock_settings:
            mock_settings.llm_timeout_seconds = 0.01
            with pytest.raises(LLMServiceUnavailableError, match="timed out"):
                await llm_invoke(llm, _MESSAGES, _CONFIG)


class TestLlmInvokeWithRetry:
    async def test_succeeds_on_first_attempt(self) -> None:
        llm = _make_llm()
        result = await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)
        assert result == _RESPONSE
        llm.ainvoke.assert_awaited_once()

    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        llm = MagicMock()
        llm.metadata = None
        llm.ainvoke = AsyncMock(
            side_effect=[
                RateLimitError("429", llm_provider="test", model="test"),
                _RESPONSE,
            ]
        )
        with (
            patch("app.graph.nodes._llm_invoke.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.llm_timeout_seconds = 60.0
            mock_settings.llm_max_retries = 3
            result = await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)
        assert result == _RESPONSE
        assert llm.ainvoke.await_count == 2

    async def test_exhausts_retries_and_raises(self) -> None:
        llm = _make_llm(side_effect=ServiceUnavailableError("503", llm_provider="test", model="test"))
        with (
            patch("app.graph.nodes._llm_invoke.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.llm_timeout_seconds = 60.0
            mock_settings.llm_max_retries = 2
            with pytest.raises(LLMServiceUnavailableError):
                await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)
        assert llm.ainvoke.await_count == 3  # 1 attempt + 2 retries

    async def test_non_retryable_error_raises_immediately(self) -> None:
        llm = _make_llm(side_effect=APIError(500, "internal error", llm_provider="test", model="test"))
        with patch("app.graph.nodes._llm_invoke.settings") as mock_settings:
            mock_settings.llm_timeout_seconds = 60.0
            mock_settings.llm_max_retries = 3
            with pytest.raises(LLMServiceError):
                await llm_invoke_with_retry(llm, _MESSAGES, _CONFIG)
        llm.ainvoke.assert_awaited_once()
