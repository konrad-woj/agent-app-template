"""Entry point: uv run python -m app.mcp starts the MCP tool server."""

import uvicorn

from app.config import settings
from app.mcp.server import mcp

uvicorn.run(mcp.http_app(), host=settings.mcp_host, port=settings.mcp_port)
