"""Tests for API key authentication (4.1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from app.auth import verify_api_key
from app.dependencies import get_graph
from app.routers import health_router, router

# Separate app with auth applied to v1 routes (mirrors production main.py setup).
_auth_app = FastAPI()
_auth_app.include_router(health_router)
_auth_app.include_router(router, dependencies=[Depends(verify_api_key)])

_VALID_KEY = "test-api-key-123"


@pytest.fixture
def mock_graph() -> MagicMock:
    g = MagicMock()
    g.aget_state = AsyncMock(return_value=MagicMock(next=(), values={}))
    g.ainvoke = AsyncMock(return_value={"status": "done"})
    return g


@pytest.fixture
async def auth_client(mock_graph: MagicMock, monkeypatch):
    from app import config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "api_key", _VALID_KEY)
    _auth_app.dependency_overrides[get_graph] = lambda: mock_graph
    async with AsyncClient(transport=ASGITransport(app=_auth_app), base_url="http://test") as ac:
        yield ac
    _auth_app.dependency_overrides.clear()


class TestHealthNoAuth:
    async def test_health_returns_200_without_api_key(self, auth_client: AsyncClient) -> None:
        response = await auth_client.get("/health")
        assert response.status_code == 200

    async def test_health_returns_200_with_api_key(self, auth_client: AsyncClient) -> None:
        response = await auth_client.get("/health", headers={"X-API-Key": _VALID_KEY})
        assert response.status_code == 200


class TestProtectedRoutes:
    async def test_missing_key_returns_401(self, auth_client: AsyncClient) -> None:
        response = await auth_client.post("/v1/chat", json={"thread_id": "t-1", "message": "hello"})
        assert response.status_code == 401

    async def test_wrong_key_returns_401(self, auth_client: AsyncClient) -> None:
        response = await auth_client.post(
            "/v1/chat",
            json={"thread_id": "t-1", "message": "hello"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    async def test_correct_key_returns_200(self, auth_client: AsyncClient) -> None:
        response = await auth_client.post(
            "/v1/chat",
            json={"thread_id": "t-1", "message": "hello"},
            headers={"X-API-Key": _VALID_KEY},
        )
        assert response.status_code == 200


class TestAuthDisabledWhenKeyUnset:
    async def test_no_key_required_when_api_key_unset(self, mock_graph: MagicMock, monkeypatch) -> None:
        from app import config as cfg_module

        monkeypatch.setattr(cfg_module.settings, "api_key", None)
        _auth_app.dependency_overrides[get_graph] = lambda: mock_graph
        async with AsyncClient(transport=ASGITransport(app=_auth_app), base_url="http://test") as ac:
            response = await ac.post("/v1/chat", json={"thread_id": "t-1", "message": "hello"})
        _auth_app.dependency_overrides.clear()
        # 200 even without X-API-Key header because api_key is None.
        assert response.status_code == 200
