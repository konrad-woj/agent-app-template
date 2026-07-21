"""Shared helpers for reading values out of AgentState's messages list."""

from langchain_core.messages import AnyMessage, HumanMessage


def get_last_human_text(messages: list[AnyMessage]) -> str:
    """Return the most recent HumanMessage's content as a string, or "" if none exists."""
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    return str(last_human.content) if last_human else ""
