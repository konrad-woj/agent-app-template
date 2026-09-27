"""Shared helpers for reading values out of AgentState's messages list."""

from langchain_core.messages import AnyMessage, HumanMessage


def get_last_human_message(messages: list[AnyMessage]) -> HumanMessage | None:
    """Return the most recent HumanMessage object, or None if none exists."""
    return next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)


def get_last_human_text(messages: list[AnyMessage]) -> str:
    """Return the most recent HumanMessage's content as a string, or "" if none exists."""
    last_human = get_last_human_message(messages)
    return str(last_human.content) if last_human else ""
