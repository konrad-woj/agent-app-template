class LLMError(Exception):
    """Base class for LLM-related errors."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate-limit (429) response."""


class LLMServiceUnavailableError(LLMError):
    """Raised when the LLM provider is unreachable or returns 503."""


class LLMServiceError(LLMError):
    """Raised for unexpected LLM provider errors (5xx other than 503)."""


class GuardTimeoutError(Exception):
    """Raised when a GLiGuard classification call exceeds its timeout."""
