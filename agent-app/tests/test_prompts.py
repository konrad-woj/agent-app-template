"""Tests for externalized prompt files under app/prompts/en/.

Walks every .md file to catch empty files and .human.md slot/brace mistakes,
then spot-checks that the loader returns the exact text of the constants that
used to be hardcoded in each node module.
"""

import string
from pathlib import Path

import pytest

from app.graph.nodes.input_guard import _TOPIC_CHECK_PROMPT
from app.graph.nodes.output_guard import _SAFE_FALLBACK
from app.graph.nodes.planner import _PLAN_GUARD_PROMPT, _PLAN_SYSTEM_PROMPT
from app.graph.nodes.subgraphs.reflection import _CRITIC_PROMPT, _REFINER_PROMPT
from app.graph.nodes.subgraphs.verification import _VERIFY_PROMPT
from app.graph.nodes.writer import _SYSTEM_PROMPT
from app.prompts.loader import load_system, load_text, render_human

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "app" / "prompts" / "en"

_SYSTEM_PROMPT_CASES = [
    ("input_guard", "topic_check", _TOPIC_CHECK_PROMPT),
    ("planner", "plan", _PLAN_SYSTEM_PROMPT),
    ("planner", "plan_guard", _PLAN_GUARD_PROMPT),
    ("writer", "draft", _SYSTEM_PROMPT),
    ("reflection", "critic", _CRITIC_PROMPT),
    ("reflection", "refiner", _REFINER_PROMPT),
    ("verification", "verify", _VERIFY_PROMPT),
]

_HUMAN_TEMPLATE_CASES = [
    ("planner", "plan_guard", {"plan": "X"}, "Research plan:\nX"),
    ("writer", "draft", {"question": "Q", "plan_summary": "P"}, "Question: Q\n\nResearch plan:\nP"),
    ("reflection", "critic", {"draft": "D"}, "Draft answer:\nD"),
    ("reflection", "refiner", {"draft": "D", "critique": "C"}, "Draft:\nD\n\nCritique:\nC"),
    ("verification", "verify", {"claim": "C", "evidence": "E"}, "Claim: C\n\nEvidence:\nE"),
]


def _all_prompt_files() -> list[Path]:
    return [p for p in _PROMPTS_ROOT.rglob("*.md") if p.is_file()]


def test_no_prompt_file_is_empty() -> None:
    files = _all_prompt_files()
    assert files, "expected at least one prompt file under app/prompts/en/"
    for path in files:
        assert path.read_text(encoding="utf-8").strip(), f"{path} is empty"


def test_every_human_template_formats_with_its_declared_slots() -> None:
    human_files = [p for p in _all_prompt_files() if p.name.endswith(".human.md")]
    assert human_files, "expected at least one .human.md template"
    for path in human_files:
        template = path.read_text(encoding="utf-8")
        slots = {field for _, field, _, _ in string.Formatter().parse(template) if field}
        template.format(**dict.fromkeys(slots, "x"))


@pytest.mark.parametrize(("node", "name", "expected"), _SYSTEM_PROMPT_CASES)
def test_system_prompt_matches_node_constant(node: str, name: str, expected: str) -> None:
    assert load_system(node, name) == expected


@pytest.mark.parametrize(("node", "name", "kwargs", "expected"), _HUMAN_TEMPLATE_CASES)
def test_human_template_renders(node: str, name: str, kwargs: dict[str, str], expected: str) -> None:
    assert render_human(node, name, **kwargs) == expected


def test_output_guard_safe_fallback() -> None:
    assert load_text("output_guard", "safe_fallback") == _SAFE_FALLBACK
