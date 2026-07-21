"""Rate limiting via slowapi.

The limiter is enabled only when AGENT_RATE_LIMIT is set (e.g. "20/minute").
When unset, the limiter is disabled and all endpoints are unrestricted.

Keyed on client IP via get_remote_address.  Behind a reverse proxy, set the
trusted FORWARDED or X-Forwarded-For headers so get_remote_address reads the
real client IP rather than the proxy address.

Usage in main.py:
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from app.rate_limit import limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Usage on an endpoint (request: Request must be a parameter):
    from app.rate_limit import limiter
    @router.post("/v1/chat")
    @limiter.limit(settings.rate_limit or "10000/minute")
    async def chat(request: Request, ...) -> ...:
        ...
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit is not None,
)
