"""Shared layer-1 (regex sanitiser) + layer-2 (GLiGuard injection check) pass.

Used by both input_guard and resume_guard, which apply the same two checks
to different messages (the first user turn vs. a resume message) with only
their log-key prefix and user-facing wording differing.
"""

from logger import get_logger

from app.config import settings
from app.graph.nodes._prompt_utils import sanitize_user_text
from app.guards.gliguard import GLiGuardClient

logger = get_logger(__name__)


async def run_sanitize_and_injection_check(
    gliguard: GLiGuardClient,
    raw_text: str,
    node_name: str,
    input_label: str,
    injection_fallback_reason: str,
) -> str | dict:
    """Run layer-1 regex sanitisation then layer-2 GLiGuard injection/jailbreak check.

    Args:
        gliguard: Loaded GLiGuardClient.
        raw_text: Raw user or resume text to check.
        node_name: Log-key prefix, e.g. "input_guard" or "resume_guard".
        input_label: Label used in the sanitiser-rejection message, e.g.
            "Input" or "Resume message".
        injection_fallback_reason: Reason used when GLiGuard blocks without one.

    Returns:
        The sanitised text on success, or a ``{"status": "blocked", ...}`` dict
        ready to return directly from the calling node on failure.
    """
    try:
        clean_text = sanitize_user_text(raw_text)
    except ValueError as exc:
        logger.info(f"{node_name}.layer1_blocked", reason=str(exc))
        return {"status": "blocked", "guard_reason": f"{input_label} rejected by sanitiser: {exc}"}
    logger.info(f"{node_name}.layer1_passed")

    guard_result = await gliguard.acheck_input(clean_text, settings.guard_timeout_seconds)
    if guard_result.blocked:
        logger.info(f"{node_name}.layer2_blocked", reason=guard_result.reason)
        return {"status": "blocked", "guard_reason": guard_result.reason or injection_fallback_reason}
    logger.info(f"{node_name}.layer2_passed")

    return clean_text
