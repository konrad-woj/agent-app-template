"""Writer node: drafts a final answer and extracts verifiable claims.

Receives the research context from react_researcher (via state["messages"]) and
the research plan. Produces draft_answer (prose) and claims (list of specific
verifiable facts), which flow into verify_subgraph and then reflection_subgraph.

The LLM is prompted to return JSON with two keys: "answer" and "claims".
On parse failure the raw content is used as draft_answer and claims is empty —
verify_subgraph fans out to zero branches, which is valid.
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from logger import get_logger
from pydantic import BaseModel

from app.graph.nodes._dead_letter import with_dead_letter
from app.graph.nodes._llm_invoke import llm_invoke_with_retry, parse_structured
from app.graph.nodes._messages import get_last_human_text
from app.prompts.loader import load_system, render_human

logger = get_logger(__name__)

# Hard ceiling on claims forwarded to verify_subgraph: each claim spawns a parallel
# Send branch (tool call + LLM call), so this bounds fan-out width regardless of
# what the writer LLM actually returns (the "3-5" in the prompt is not enforced by it).
_MAX_CLAIMS = 5

_SYSTEM_PROMPT = load_system("writer", "draft")


class _WriterOutput(BaseModel):
    answer: str
    claims: list[str]


def make_writer_node(llm: BaseChatModel) -> Callable:
    """Return an async writer node bound to the given LLM."""

    @with_dead_letter("writer")
    async def writer(state: "AgentState", config: RunnableConfig) -> dict:  # type: ignore[name-defined]  # noqa: F821
        question = get_last_human_text(state["messages"])
        plan: list[str] = state.get("plan", [])  # type: ignore[assignment]
        logger.info(
            "writer.inputs",
            question_length=len(question),
            plan_step_count=len(plan),
            message_count=len(state["messages"]),
        )
        plan_summary = "\n".join(plan)
        context = render_human("writer", "draft", question=question, plan_summary=plan_summary)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response = await llm_invoke_with_retry(llm, messages, config)
        raw = str(response.content)
        parsed = parse_structured(raw, _WriterOutput)
        if parsed is not None:
            draft = parsed.answer
            claims = parsed.claims
        else:
            logger.warning("writer.parse_failed", raw_length=len(raw))
            draft = raw
            claims = []
        if len(claims) > _MAX_CLAIMS:
            logger.warning("writer.claims_truncated", claim_count=len(claims), max_claims=_MAX_CLAIMS)
            claims = claims[:_MAX_CLAIMS]
        logger.info("writer.draft_produced", draft_length=len(draft), claim_count=len(claims))
        return {"draft_answer": draft, "claims": claims, "status": "writing"}

    return writer
