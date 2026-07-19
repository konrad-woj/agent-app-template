"""API key authentication dependency.

Authentication is enabled only when AGENT_API_KEY is set in the environment.
When AGENT_API_KEY is unset (default), all requests are allowed — this preserves
backward compatibility for local development and tests.

Usage in main.py:
    from app.auth import verify_api_key
    app.include_router(router, dependencies=[Depends(verify_api_key)])
"""

import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Validate the X-API-Key header against AGENT_API_KEY.

    Short-circuits (allows all requests) when AGENT_API_KEY is not configured.

    Raises:
        HTTPException: 401 when AGENT_API_KEY is set and the header is missing
            or does not match.
    """
    if settings.api_key is None:
        return
    if api_key is None or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
