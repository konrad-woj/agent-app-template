import uvicorn

from app.config import settings

uvicorn.run(
    "app.main:app",
    host=settings.app_host,
    port=settings.app_port,
    reload=False,
    log_level="info",
    timeout_graceful_shutdown=settings.graceful_shutdown_timeout_seconds,
)
