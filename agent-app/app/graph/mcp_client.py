"""MCP client factory: connects to the MCP tool server and returns LangChain tools.

Tools are loaded once at app startup in lifespan() and injected into the
react_researcher node via compile_graph(). This avoids reconnecting on every
node invocation just to *discover* tools. It does NOT mean a single MCP
connection is held open for the app's lifetime: MultiServerMCPClient.get_tools()
opens a new session per tool call under the hood ("A new session will be
created for each tool call" per its own docstring), so each tool invocation
still pays its own connection cost regardless of when the tools were loaded.

Note: MultiServerMCPClient does NOT support async context manager as of 0.1.x.
Instantiate directly and await get_tools() — do not use async with.

Example usage (in lifespan):
    mcp_tools = await load_mcp_tools(settings.mcp_server_url)
    app.state.graph = compile_graph(checkpointer, build_llm(), mcp_tools)
"""

import asyncio

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from logger import get_logger

from app.config import settings

logger = get_logger(__name__)


async def load_mcp_tools(server_url: str, retries: int = 5, delay: float = 2.0) -> list[BaseTool]:
    """Connect to the MCP server and return all exposed tools as LangChain BaseTool instances.

    Retries up to `retries` times with `delay` seconds between attempts to handle
    race conditions when the MCP server process is still starting. Each attempt
    is bounded by settings.mcp_connect_timeout_seconds so a hung connection can't
    stall app startup indefinitely.

    Args:
        server_url: Base URL of the fastmcp server (e.g. "http://localhost:8001").
        retries: Number of connection attempts before raising.
        delay: Seconds to wait between attempts.
    """
    last_exc: Exception = RuntimeError("retries must be at least 1")
    for attempt in range(max(retries, 0)):
        try:
            client = MultiServerMCPClient({"research": {"url": server_url, "transport": "streamable_http"}})
            return await asyncio.wait_for(client.get_tools(), timeout=settings.mcp_connect_timeout_seconds)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                logger.warning(
                    "MCP server not ready, retrying",
                    attempt=attempt + 1,
                    retries=retries,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
    raise last_exc
