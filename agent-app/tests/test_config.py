"""Tests for app/config.py Settings — env var overrides for critical fields."""

from unittest.mock import patch

from app.config import Settings, settings
from app.main import _warn_if_insecure_defaults


class TestGracefulShutdownTimeout:
    def test_default_is_below_typical_k8s_grace_period(self) -> None:
        """Default must stay below the common 30s terminationGracePeriodSeconds so the
        process always exits cleanly on its own rather than being SIGKILLed."""
        settings = Settings()
        assert 0 < settings.graceful_shutdown_timeout_seconds < 30

    def test_overridable_via_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENT_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS", "10")
        settings = Settings()
        assert settings.graceful_shutdown_timeout_seconds == 10


class TestWarnIfInsecureDefaults:
    """AGENT_API_KEY / AGENT_RATE_LIMIT are deliberately optional (local dev / tests),
    but a real deployment left with either unset must get a loud startup warning."""

    def test_warns_when_api_key_unset(self) -> None:
        with (
            patch.object(settings, "api_key", None),
            patch.object(settings, "rate_limit", "20/minute"),
            patch("app.main.logger") as mock_logger,
        ):
            _warn_if_insecure_defaults()
        assert mock_logger.warning.call_count == 1
        assert "AGENT_API_KEY" in mock_logger.warning.call_args[0][0]

    def test_warns_when_rate_limit_unset(self) -> None:
        with (
            patch.object(settings, "api_key", "secret"),
            patch.object(settings, "rate_limit", None),
            patch("app.main.logger") as mock_logger,
        ):
            _warn_if_insecure_defaults()
        assert mock_logger.warning.call_count == 1
        assert "AGENT_RATE_LIMIT" in mock_logger.warning.call_args[0][0]

    def test_no_warning_when_both_configured(self) -> None:
        with (
            patch.object(settings, "api_key", "secret"),
            patch.object(settings, "rate_limit", "20/minute"),
            patch("app.main.logger") as mock_logger,
        ):
            _warn_if_insecure_defaults()
        mock_logger.warning.assert_not_called()

    def test_warns_twice_when_both_unset(self) -> None:
        with (
            patch.object(settings, "api_key", None),
            patch.object(settings, "rate_limit", None),
            patch("app.main.logger") as mock_logger,
        ):
            _warn_if_insecure_defaults()
        assert mock_logger.warning.call_count == 2
