"""GLiGuard wrapper for input/output guardrail classification and PII redaction.

Uses fastino/gliguard-LLMGuardrails-300M via the gliner2 library (GLiNER2 backbone).
The model is loaded lazily via load() — import-time has no side effects so tests
can inject a mock without triggering a HuggingFace model download.

Typical lifespan usage:
    gliguard = GLiGuardClient(settings.guard_model, settings.guard_device)
    gliguard.load()
    app.state.gliguard = gliguard

Test usage (no model needed):
    gliguard = MagicMock(spec=GLiGuardClient)
    gliguard.check_input.return_value = GuardResult(blocked=False)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from logger import get_logger

from app.exceptions import GuardTimeoutError

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Task config for injection/jailbreak classification.
_INPUT_SAFETY_TASKS: dict[str, list[str]] = {
    "prompt_safety": ["safe", "unsafe"],
}

# Entity labels for PII detection in generated output.
_PII_LABELS: list[str] = [
    "email",
    "phone number",
    "credit card number",
    "social security number",
    "api key",
    "ip address",
]


@dataclass
class Span:
    """A detected entity span within a text string."""

    text: str
    entity_type: str
    start: int
    end: int


@dataclass
class GuardResult:
    """Result returned by GLiGuardClient check methods.

    For check_input: blocked=True when injection/jailbreak is detected.
    For check_output: blocked is always False (PII is redacted, not blocked);
        flagged_spans carries the PII character ranges for redaction.
    """

    blocked: bool
    reason: str | None = None
    flagged_spans: list[Span] = field(default_factory=list)


def redact(text: str, spans: list[Span]) -> str:
    """Replace PII spans in text with [REDACTED:<entity_type>] tokens.

    Processes spans in reverse start-offset order so earlier replacements
    don't shift the character indices of later ones.

    Args:
        text: Original text containing PII.
        spans: Detected PII spans from GLiGuardClient.check_output.

    Returns:
        Text with each span replaced by [REDACTED:<entity_type>].

    Example:
        >>> redact("Call 555-1234 or email foo@bar.com", spans)
        'Call [REDACTED:phone number] or email [REDACTED:email]'
    """
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        replacement = f"[REDACTED:{span.entity_type}]"
        text = text[: span.start] + replacement + text[span.end :]
    return text


def _find_entity_spans(text: str, entities: dict[str, list[str]]) -> list[Span]:
    """Build Span objects from extract_entities output by locating each entity string."""
    spans: list[Span] = []
    for entity_type, values in entities.items():
        for value in values:
            start = 0
            while True:
                idx = text.find(value, start)
                if idx == -1:
                    break
                spans.append(Span(text=value, entity_type=entity_type, start=idx, end=idx + len(value)))
                start = idx + len(value)
    return spans


def _extract_entities_dict(result: dict) -> dict[str, list[str]]:
    """Normalise extract_entities output to {label: [matched_text, ...]}."""
    if "entities" in result:
        return result["entities"]
    return {k: v for k, v in result.items() if isinstance(v, list)}


class GLiGuardClient:
    """Thin wrapper around GLiNER2 (fastino/gliguard-LLMGuardrails-300M).

    The model is NOT loaded at __init__ time. Call load() from the FastAPI
    lifespan before the first request arrives.

    Args:
        model_name: HuggingFace model identifier.
        device: Inference device: 'cpu', 'cuda', or 'mps'.
        max_concurrency: Maximum concurrent inference calls through acheck_input/
            acheck_output. All requests share this one loaded model instance, so
            unbounded concurrency means unbounded concurrent forward passes —
            each with its own activation memory — which can exhaust host RAM or,
            worse, GPU VRAM under load. Excess calls queue on a semaphore instead.
    """

    def __init__(self, model_name: str, device: str = "cpu", max_concurrency: int = 4) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None  # populated by load()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def loaded(self) -> bool:
        """Whether load() has completed and the model is ready for inference."""
        return self._model is not None

    def load(self) -> None:
        """Download (if needed) and load the GLiNER2 model into memory."""
        from gliner2 import GLiNER2  # lazy import — not needed at test time

        logger.info("Loading GLiGuard model", model=self._model_name, device=self._device)
        self._model = GLiNER2.from_pretrained(self._model_name)
        self._model.to(self._device)
        logger.info("GLiGuard model loaded")

    async def aload(self, timeout: float, retries: int = 3, delay: float = 5.0) -> None:
        """Async, timeout-and-retry-bounded wrapper around load(), for use at app startup.

        The HF download can hang on a network blip or transiently fail; without a
        bound, a stuck download hangs app startup indefinitely with no diagnostic
        beyond whatever eventually kills the process (e.g. a k8s startupProbe
        timeout, with no log line explaining why). Runs load() in a worker thread
        so it never blocks the event loop while (re)trying.

        Args:
            timeout: Per-attempt timeout in seconds.
            retries: Number of attempts before raising the last error.
            delay: Seconds to wait between attempts.

        Raises:
            The last exception encountered (TimeoutError on the final attempt's
            timeout, or whatever load() raised) once retries are exhausted.
        """
        last_exc: Exception = RuntimeError("retries must be at least 1")
        for attempt in range(max(retries, 0)):
            try:
                await asyncio.wait_for(asyncio.to_thread(self.load), timeout=timeout)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1:
                    logger.warning(
                        "GLiGuard model load failed, retrying",
                        attempt=attempt + 1,
                        retries=retries,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
        logger.error("GLiGuard model load failed after all retries", retries=retries, error=str(last_exc))
        raise last_exc

    def check_input(self, text: str) -> GuardResult:
        """Classify user input for prompt injection and jailbreak attempts.

        Args:
            text: Sanitised user text (post layer-1 regex pass).

        Returns:
            GuardResult with blocked=True when injection/jailbreak is detected.
        """
        if self._model is None:
            raise RuntimeError("GLiGuardClient.load() must be called before check_input().")

        result: dict = self._model.classify_text(text, _INPUT_SAFETY_TASKS, threshold=0.5)
        verdict = result.get("prompt_safety", "safe")
        if isinstance(verdict, list):
            verdict = verdict[0] if verdict else "safe"

        blocked = verdict != "safe"
        return GuardResult(
            blocked=blocked,
            reason=f"GLiGuard detected unsafe content: {verdict}" if blocked else None,
        )

    def check_output(self, text: str) -> GuardResult:
        """Detect PII spans in generated output for redaction.

        Does NOT block — always returns blocked=False with flagged_spans
        populated so the caller can redact before returning to the user.

        Args:
            text: Draft final answer before returning to caller.

        Returns:
            GuardResult with blocked=False and flagged_spans listing any PII found.
        """
        if self._model is None:
            raise RuntimeError("GLiGuardClient.load() must be called before check_output().")

        result: dict = self._model.extract_entities(text, _PII_LABELS)
        entities = _extract_entities_dict(result)
        spans = _find_entity_spans(text, entities)

        if spans:
            entity_types = list({s.entity_type for s in spans})
            logger.warning("PII detected in output — redacting", entity_types=entity_types, span_count=len(spans))

        return GuardResult(blocked=False, flagged_spans=spans)

    async def acheck_input(self, text: str, timeout: float) -> GuardResult:
        """Async, timeout-bounded wrapper around check_input.

        Runs the classification call in a worker thread so it never blocks the
        event loop, and bounds it with `timeout` so a hung/slow model can't stall
        a request indefinitely.

        Raises:
            GuardTimeoutError: if the call exceeds `timeout` seconds.
        """
        async with self._semaphore:
            try:
                return await asyncio.wait_for(asyncio.to_thread(self.check_input, text), timeout=timeout)
            except TimeoutError as exc:
                raise GuardTimeoutError(f"GLiGuard check_input timed out after {timeout}s") from exc

    async def acheck_output(self, text: str, timeout: float) -> GuardResult:
        """Async, timeout-bounded wrapper around check_output. See acheck_input."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(asyncio.to_thread(self.check_output, text), timeout=timeout)
            except TimeoutError as exc:
                raise GuardTimeoutError(f"GLiGuard check_output timed out after {timeout}s") from exc
