"""Tests for app/__main__.py — the production uvicorn entrypoint."""

import importlib
import sys
from unittest.mock import patch


class TestEntrypoint:
    def test_passes_graceful_shutdown_timeout_from_settings(self) -> None:
        with patch("uvicorn.run") as mock_run:
            sys.modules.pop("app.__main__", None)
            importlib.import_module("app.__main__")
        _, kwargs = mock_run.call_args
        from app.config import settings

        assert kwargs["timeout_graceful_shutdown"] == settings.graceful_shutdown_timeout_seconds

    def test_passes_host_and_port_from_settings(self) -> None:
        with patch("uvicorn.run") as mock_run:
            sys.modules.pop("app.__main__", None)
            importlib.import_module("app.__main__")
        _, kwargs = mock_run.call_args
        from app.config import settings

        assert kwargs["host"] == settings.app_host
        assert kwargs["port"] == settings.app_port
