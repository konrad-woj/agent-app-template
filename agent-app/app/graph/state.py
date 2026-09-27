"""Shared state definition for the research-assistant graph.

LangGraph passes the full state dict into every node and merges each node's
returned dict back into state after the node completes.  Fields that carry
a *reducer* annotation are merged with that function instead of overwritten:

  messages: Annotated[list[AnyMessage], add_messages]

The ``add_messages`` reducer appends the new messages list to the existing
one, so each node only returns the messages it produced — not the whole
history.  It also assigns a stable ``id`` to any message that lacks one, and
lets a node replace or delete a specific past message by returning a
``RemoveMessage(id=...)`` alongside its replacement (see input_guard and
resume_guard, which use this to swap the raw user message for its sanitised
form once cleaned).

All other fields (plain TypedDict entries) are last-write-wins: whichever node
writes them last wins. That is intentional for scalar fields like ``status``.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.graph.nodes._dead_letter import DeadLetterInfo


class AgentState(TypedDict):
    # add_messages reducer: nodes append to this list by default, and may
    # replace/delete a specific message via RemoveMessage(id=...).
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]  # planner output; each entry is one research step
    plan_approved: bool
    claims: list[str]  # verifiable factual claims extracted by writer
    verification_results: list[dict]  # per-claim results from verify_subgraph
    react_steps: int  # incremented each ReAct iteration (observability)
    draft_answer: str  # writer output before reflection
    reflection_attempts: int
    reflection_passed: bool
    final_answer: str  # output_guard-approved answer returned to the caller
    # Lifecycle: "planning" → "researching" → "writing" → "verifying" →
    #            "reflecting" → "done" | "aborted" | "blocked" | "dead_lettered"
    status: str
    guard_reason: str | None  # set when input_guard or output_guard blocks
    dead_letter: DeadLetterInfo | None  # set by with_dead_letter on unhandled exception
