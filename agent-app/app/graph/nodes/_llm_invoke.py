"""Centralised async LLM wrapper with timeout enforcement and retry logic.

All node functions that call the LLM must use llm_invoke_with_retry instead
of calling llm.ainvoke directly. This ensures:

  - Every call is bounded by AGENT_LLM_TIMEOUT_SECONDS (default 60s), or a
    per-node override stored in llm.metadata["timeout_seconds"].
    asyncio.TimeoutError is translated to LLMServiceUnavailableError so the
    dead-letter decorator can handle it uniformly.

  - Transient errors (rate limit, 5xx / service unavailable) are retried up to
    AGENT_LLM_MAX_RETRIES times (default 3), or a per-node override stored in
    llm.metadata["max_retries"], with exponential backoff capped at 30s.
    Non-transient errors (LLMError base, other exceptions) are re-raised immediately.

Example usage:
    response = await llm_invoke_with_retry(llm, state["messages"], config)

Node-level overrides:
    from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm
    llm = build_llm(NodeLLMConfig(thinking=True, timeout_seconds=120))
"""

import asyncio
from dataclasses import dataclass

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
from litellm.exceptions import APIConnectionError, APIError, RateLimitError, ServiceUnavailableError
from logger import get_logger
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.exceptions import LLMError, LLMRateLimitError, LLMServiceError, LLMServiceUnavailableError

logger = get_logger(__name__)


@dataclass
class NodeLLMConfig:
    """Per-node LLM overrides. Any field left None falls back to the global Settings value."""

    model: str | None = None
    temperature: float | None = None
    thinking: bool | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None


def build_llm(override: NodeLLMConfig | None = None) -> ChatLiteLLM:
    """Construct a LangChain LLM, merging an optional per-node override onto global settings.

    Timeout and max_retries are stored in llm.metadata so llm_invoke /
    llm_invoke_with_retry can read them without changing their signatures.

    Args:
        override: Per-node config; None means use all global settings defaults.
    """
    cfg = override or NodeLLMConfig()
    timeout = cfg.timeout_seconds if cfg.timeout_seconds is not None else settings.llm_timeout_seconds
    max_retries = cfg.max_retries if cfg.max_retries is not None else settings.llm_max_retries
    thinking = cfg.thinking if cfg.thinking is not None else settings.llm_thinking
    return ChatLiteLLM(
        model=cfg.model or settings.llm_model,
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=cfg.temperature,
        streaming=True,  # required for astream_events to emit on_chat_model_stream chunks; ainvoke still returns a complete aggregated message so the non-streaming endpoint is unaffected
        model_kwargs={"enable_thinking": thinking},
        metadata={"timeout_seconds": timeout, "max_retries": max_retries},
    )


def _translate_exception(exc: Exception) -> LLMError | None:
    """Map known provider-level exceptions to the internal LLMError hierarchy.

    Returns None for unknown exceptions so they propagate unmasked.
    """
    if isinstance(exc, RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, (ServiceUnavailableError, APIConnectionError)):
        return LLMServiceUnavailableError(str(exc))
    if isinstance(exc, APIError):
        return LLMServiceError(str(exc))
    if isinstance(exc, httpx.ConnectError):
        return LLMServiceUnavailableError(str(exc))
    return None


async def llm_invoke(
    llm: BaseChatModel,
    messages: list[AnyMessage],
    config: RunnableConfig,
) -> AnyMessage:
    """Single LLM call wrapped in a timeout.

    Reads timeout_seconds from llm.metadata if set; falls back to settings.

    Raises:
        LLMServiceUnavailableError: on timeout or connection failure.
        LLMRateLimitError: on 429 responses.
        LLMServiceError: on unexpected provider errors.
    """
    timeout = (llm.metadata or {}).get("timeout_seconds", settings.llm_timeout_seconds)
    try:
        return await asyncio.wait_for(
            llm.ainvoke(messages, config),
            timeout=timeout,
        )
    except TimeoutError as err:
        raise LLMServiceUnavailableError(f"LLM call timed out after {timeout}s") from err
    except LLMError:
        raise
    except Exception as err:
        translated = _translate_exception(err)
        if translated is not None:
            raise translated from err
        raise


async def llm_invoke_with_retry(
    llm: BaseChatModel,
    messages: list[AnyMessage],
    config: RunnableConfig,
) -> AnyMessage:
    """LLM call with exponential-backoff retry for transient errors.

    Reads max_retries from llm.metadata if set; falls back to settings.
    Retries on LLMRateLimitError or LLMServiceUnavailableError. All other
    LLMError subclasses are re-raised immediately without retrying.

    Raises:
        LLMRateLimitError | LLMServiceUnavailableError: after all retries exhausted.
        LLMServiceError: immediately on non-retryable provider error.
    """
    _retryable = (LLMRateLimitError, LLMServiceUnavailableError)
    max_retries = (llm.metadata or {}).get("max_retries", settings.llm_max_retries)
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await llm_invoke(llm, messages, config)
        except _retryable as err:
            last_err = err
            if attempt < max_retries:
                wait = min(2**attempt, 30)
                logger.warning(
                    "LLM transient error, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(err),
                )
                await asyncio.sleep(wait)
        except LLMError:
            raise

    raise last_err  # type: ignore[misc]


def parse_structured[T: BaseModel](raw: str, schema: type[T]) -> T | None:
    """Parse an LLM response's raw text content as `schema`; return None on parse failure.

    Every node that prompts an LLM for structured JSON follows the same
    call-then-parse shape; this factors out just the parsing step so each
    caller keeps its own fallback behaviour for the None case.
    """
    try:
        return schema.model_validate_json(raw)
    except (ValidationError, ValueError):
        return None
