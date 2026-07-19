"""Loads externalized prompt files under app/prompts/<locale>/<node>/<name>.

System prompts are 100% static text (some contain literal JSON braces in their
"respond with JSON" instructions) and are always read raw via .read_text() —
never passed through .format(). Only .human.md files, which have genuine
{slot} placeholders and no literal braces, are rendered with .format(**kwargs).

Example:
    _TOPIC_CHECK_PROMPT = load_system("input_guard", "topic_check")
    context = render_human("writer", "draft", question=question, plan_summary=plan_summary)
"""

import importlib.resources
from functools import cache

from app.config import settings

_PACKAGE = "app.prompts"


@cache
def _read(locale: str, node: str, filename: str) -> str:
    path = importlib.resources.files(_PACKAGE) / locale / node / filename
    return path.read_text(encoding="utf-8")


def load_system(node: str, name: str, *, locale: str | None = None) -> str:
    """Read a system prompt file raw. Never call .format() on the result."""
    return _read(locale or settings.locale, node, f"{name}.system.md")


def render_human(node: str, name: str, *, locale: str | None = None, **kwargs: object) -> str:
    """Read a human message template and render its {slot} placeholders."""
    template = _read(locale or settings.locale, node, f"{name}.human.md")
    return template.format(**kwargs)


def load_text(node: str, name: str, *, locale: str | None = None) -> str:
    """Read a plain, non-templated text file raw (e.g. a user-facing fallback message)."""
    return _read(locale or settings.locale, node, f"{name}.md")
