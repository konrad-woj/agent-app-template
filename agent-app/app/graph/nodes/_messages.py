"""Shared helpers for reading values out of AgentState's messages list."""

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage


def get_last_human_message(messages: list[AnyMessage]) -> HumanMessage | None:
    """Return the most recent HumanMessage object, or None if none exists."""
    return next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)


def get_turn_question(state: Mapping[str, Any]) -> str:
    """The user question for the current turn, as sanitised by input_guard.

    Not the last HumanMessage: resuming from the plan-approval interrupt appends
    the resume message ("yes, go ahead") after the question. Falls back to the
    last human text when `question` is unset (node-level tests, checkpoints
    written before the field existed).
    """
    return state.get("question") or get_last_human_text(state["messages"])


def get_last_human_text(messages: list[AnyMessage]) -> str:
    """Return the most recent HumanMessage's content as a string, or "" if none exists."""
    last_human = get_last_human_message(messages)
    return str(last_human.content) if last_human else ""
