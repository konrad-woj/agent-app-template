"""Tests for API rate limiting (4.2).

Uses a test-specific FastAPI app with a 1/minute limiter so the limit can be
exceeded in unit tests without depending on the production limit string (which
is captured at decorator-apply time and defaults to "10000/minute" in local dev).
"""

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded


def _forwarded_ip(request: Request) -> str:
    """Key function that reads X-Forwarded-For so tests can simulate multiple IPs."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


_test_limiter = Limiter(key_func=_forwarded_ip, enabled=True)


def _test_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    response = JSONResponse({"detail": f"Rate limit exceeded: {exc.detail}"}, status_code=429)
    response.headers["Retry-After"] = str(60)
    return response


_rate_app = FastAPI()
_rate_app.state.limiter = _test_limiter
_rate_app.add_exception_handler(RateLimitExceeded, _test_rate_limit_handler)  # type: ignore[arg-type]


@_rate_app.get("/limited")
@_test_limiter.limit("1/minute")
async def _limited_route(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@pytest.fixture
async def rate_client() -> AsyncGenerator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=_rate_app), base_url="http://test") as ac:
        yield ac


class TestRateLimiting:
    async def test_exceeding_limit_returns_429(self, rate_client: AsyncClient) -> None:
        # Use a unique IP per test so shared limiter state doesn't cause interference.
        r1 = await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.1"})
        assert r1.status_code == 200

        r2 = await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.1"})
        assert r2.status_code == 429

    async def test_rate_limit_response_has_retry_after_header(self, rate_client: AsyncClient) -> None:
        await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.2"})
        r2 = await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.2"})

        assert r2.status_code == 429
        assert "retry-after" in {k.lower() for k in r2.headers}

    async def test_different_ips_have_independent_counters(self, rate_client: AsyncClient) -> None:
        # Exhaust the limit for IP A.
        await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.3"})

        # IP B should still be allowed (fresh counter).
        r_b = await rate_client.get("/limited", headers={"X-Forwarded-For": "10.0.1.4"})
        assert r_b.status_code == 200


class TestRateLimitDisabledByDefault:
    async def test_limiter_disabled_when_rate_limit_unset(self) -> None:
        from app.rate_limit import limiter

        # When AGENT_RATE_LIMIT is not set (default), the limiter is disabled.
        # settings.rate_limit is None in test env, so limiter.enabled == False.
        assert not limiter.enabled
