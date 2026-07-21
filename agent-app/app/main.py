"""Agent App — FastAPI entry point.

Lifecycle:
  startup  → load MCP tools, open a pooled AsyncPostgresSaver, load GLiGuardClient, compile graph.
  shutdown → on SIGTERM, uvicorn (via app/__main__.py's timeout_graceful_shutdown, sourced from
             AGENT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS) stops accepting new connections and waits
             for in-flight requests to finish — up to that timeout — before cancelling any still
             running and invoking this module's shutdown phase, which closes the connection pool.
             Keep the timeout below the deployment's terminationGracePeriodSeconds so the process
             always exits cleanly on its own instead of being SIGKILLed mid-shutdown.

The compiled graph is stored on app.state.graph and injected into endpoints
via the get_graph() dependency (app/dependencies.py).

Quick start:
    # terminal 1: MCP server
    uv run python -m app.mcp.server

    # terminal 2: FastAPI app
    uv run python -m app
    # → http://localhost:8000/docs
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from logger import configure_logging, get_logger
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from slowapi.errors import RateLimitExceeded

from app.auth import verify_api_key
from app.config import settings
from app.graph.mcp_client import load_mcp_tools
from app.graph.nodes._llm_invoke import NodeLLMConfig
from app.graph.workflow import compile_graph
from app.guards.gliguard import GLiGuardClient
from app.rate_limit import limiter
from app.routers import health_router, router

configure_logging()
logger = get_logger(__name__)


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse(
        {"detail": f"Rate limit exceeded: {exc.detail}"},
        status_code=429,
    )
    response.headers["Retry-After"] = str(60)
    return response


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _warn_if_insecure_defaults() -> None:
    """Loudly flag opt-in security settings left at their permissive local-dev defaults.

    AGENT_API_KEY and AGENT_RATE_LIMIT are deliberately unset by default so this
    template runs out of the box for local dev and tests (see app/auth.py,
    app/rate_limit.py) — that is not a bug and must not be forced to a required
    field. But a real deployment left with either unset has authless / unlimited
    endpoints with no signal in the logs, so warn loudly at startup instead.
    """
    if settings.api_key is None:
        logger.warning(
            "AGENT_API_KEY is unset — all /v1 endpoints are unauthenticated. Set it for any non-local deployment."
        )
    if settings.rate_limit is None:
        logger.warning(
            "AGENT_RATE_LIMIT is unset — graph-invoking endpoints have no rate limit. Set it for any non-local deployment."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logger.info("Agent App starting")
    _warn_if_insecure_defaults()

    mcp_tools = await load_mcp_tools(settings.mcp_server_url)
    logger.info("MCP tools loaded", tool_count=len(mcp_tools))

    gliguard = GLiGuardClient(settings.guard_model, settings.guard_device, settings.guard_max_concurrency)
    await gliguard.aload(settings.guard_load_timeout_seconds, retries=settings.guard_load_retries)
    app.state.gliguard = gliguard
    logger.info("GLiGuard loaded", model=settings.guard_model, device=settings.guard_device)

    node_llm_configs: dict[str, NodeLLMConfig] = {
        "input_guard": NodeLLMConfig(temperature=0.0),
        "planner": NodeLLMConfig(temperature=0.0),
        "react_researcher": NodeLLMConfig(temperature=0.0),
        "writer": NodeLLMConfig(temperature=0.3),
        "verification": NodeLLMConfig(temperature=0.0),
        "reflection": NodeLLMConfig(temperature=0.2),
    }

    async with AsyncConnectionPool(
        conninfo=settings.db_uri,
        max_size=settings.db_pool_max_size,
        kwargs={"autocommit": True, "row_factory": dict_row},
        connection_class=AsyncConnection[DictRow],
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # idempotent; safe to run on every startup
        app.state.checkpointer = checkpointer  # exposed so /ready can probe DB connectivity directly
        app.state.mcp_tool_count = len(mcp_tools)  # count only; tools themselves stay closed over by the graph
        app.state.graph = compile_graph(checkpointer, mcp_tools, gliguard, node_llm_configs)
        logger.info("Graph compiled and ready")
        yield

    logger.info("Agent App shutting down")


app = FastAPI(
    title="Agent App API",
    description=(
        "LangGraph research-assistant demonstrating time-travel, interrupts, "
        "subgraphs, ReAct, MCP tools, and SSE token streaming."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# /health is unauthenticated so liveness probes work without credentials.
app.include_router(health_router)
# All v1 endpoints require API key authentication when AGENT_API_KEY is set.
app.include_router(router, dependencies=[Depends(verify_api_key)])
app.add_exception_handler(Exception, _unhandled_exception_handler)
