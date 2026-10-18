"""Node-level eval runner — Phase 5e (see PLAN.md).

Runs the Phase 5a node tasks (evals/node_tasks.py) against a real LLM, one
dataset per node (evals/datasets/node/*.yaml), and checks each item's
`expected_output`. tests/evals/test_node_tasks.py covers the same tasks with a
mocked LLM; this is the prompt/quality regression run.

Each item gets PASS, FAIL (an expectation not met), or ERROR (the task raised,
the node dead-lettered — usually the LLM is unreachable — or the node couldn't
parse the LLM's response and returned its fallback verdict). Exits 1 on any
FAIL or ERROR, or if the config selects zero items.

GLiGuard stays mocked permissively here (see evals/node_tasks.py): this tier
measures the LLM-driven decision each node makes. Guard behaviour is the
guardrail_redteam suite's job.

Usage:
    uv run python evals/run.py node_eval
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.graph.nodes import input_guard, planner
from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm
from app.graph.nodes.subgraphs import verification
from evals._results import write_run
from evals.node_tasks import NODE_TASK_REGISTRY

_EVALS_DIR = Path(__file__).parent
_DATASETS_DIR = _EVALS_DIR / "datasets"


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str
    dataset: str  # resolved relative to evals/datasets/

    @field_validator("node")
    @classmethod
    def known_node(cls, v: str) -> str:
        if v not in NODE_TASK_REGISTRY:
            raise ValueError(f"unknown node {v!r}; known: {sorted(NODE_TASK_REGISTRY)}")
        return v


class NodeEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["node_eval"] = "node_eval"
    nodes: list[NodeSpec]  # a node may appear more than once, e.g. with a second dataset
    deployment: str | None = None  # LiteLLM model id; None → AGENT_LLM_MODEL

    @model_validator(mode="after")
    def datasets_load(self) -> NodeEvalConfig:
        # Load every dataset now so a malformed one is a config error (evals/run.py exits 2)
        # before any LLM call, not a crash halfway through the run.
        for spec in self.nodes:
            load_node_dataset(spec.dataset)
        return self


# ---------------------------------------------------------------------------
# expected_output checks — one per key used in evals/datasets/node/*.yaml
# ---------------------------------------------------------------------------

type Check = Callable[[dict[str, Any], Any], str | None]  # returns a failure message, or None


def _min_plan_steps(out: dict[str, Any], n: int) -> str | None:
    steps = len(out.get("plan") or [])
    return None if steps >= n else f"plan has {steps} steps, expected ≥{n}"


def _must_reach_interrupt(out: dict[str, Any], want: bool) -> str | None:
    got = bool(out.get("reached_interrupt"))
    return None if got is want else f"reached_interrupt={got}, expected {want} (status={out.get('status')})"


def _min_draft_length(out: dict[str, Any], n: int) -> str | None:
    length = len(out.get("draft_answer") or "")
    return None if length >= n else f"draft_answer is {length} chars, expected ≥{n}"


def _min_claims(out: dict[str, Any], n: int) -> str | None:
    claims = len(out.get("claims") or [])
    return None if claims >= n else f"{claims} claims extracted, expected ≥{n}"


def _has_verdict(out: dict[str, Any], want: bool) -> str | None:
    got = all(out.get(k) is not None for k in ("supported", "confidence", "reason"))
    missing = [k for k in ("supported", "confidence", "reason") if out.get(k) is None]
    return None if got is want else f"has_verdict={got}, expected {want} (missing: {missing or 'none'})"


def _expected_status(out: dict[str, Any], want: str) -> str | None:
    got = out.get("status")
    return None if got == want else f"status={got!r}, expected {want!r} (guard_reason={out.get('guard_reason')!r})"


CHECKS: dict[str, Check] = {
    "min_plan_steps": _min_plan_steps,
    "must_reach_interrupt": _must_reach_interrupt,
    "min_draft_length": _min_draft_length,
    "min_claims": _min_claims,
    "has_verdict": _has_verdict,
    "expected_status": _expected_status,
}


def check_item(expected: dict[str, Any], output: dict[str, Any]) -> list[str]:
    """Return every failed expectation for one item (empty list → PASS)."""
    return [msg for key, want in expected.items() if (msg := CHECKS[key](output, want)) is not None]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> NodeEvalConfig:
    return NodeEvalConfig.model_validate(yaml.safe_load(path.read_text()))


def load_node_dataset(dataset: str) -> list[dict]:
    """Load one node dataset, failing fast on a malformed item or an unknown expectation.

    Every returned item has a string `id` (`#<index>` when the dataset gives none),
    so results can tell items apart.

    Raises:
        ValueError: the file is missing or unreadable, or an item is malformed.
    """
    path = _DATASETS_DIR / dataset
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{dataset}: cannot read dataset: {exc}") from exc
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"{dataset}: expected a mapping with an 'items' list")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset}: item #{i} must be a mapping, got {type(item).__name__}")
        item["id"] = str(item.get("id", f"#{i}"))
        label = item["id"]
        if "input" not in item or "expected_output" not in item:
            raise ValueError(f"{dataset}: item {label} needs both 'input' and 'expected_output'")
        if not isinstance(item["expected_output"], dict):
            raise ValueError(f"{dataset}: item {label} expected_output must be a mapping")
        unknown = set(item["expected_output"]) - CHECKS.keys()
        if unknown:
            raise ValueError(f"{dataset}: item {label} has unknown expected_output keys {sorted(unknown)}")
    return items


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


# Fallback reasons nodes return when they can't parse the LLM's response.
_PARSE_FALLBACKS = frozenset(
    {
        input_guard.PARSE_FAILED_REASON,
        planner.PLAN_GUARD_PARSE_FAILED_REASON,
        verification.PARSE_FAILED_REASON,
    }
)


@dataclass(frozen=True)
class ItemResult:
    node: str
    item_id: str
    verdict: Literal["PASS", "FAIL", "ERROR"]
    detail: str
    output: dict[str, Any]


async def _run_item(node: str, task: Callable, item: dict) -> ItemResult:
    item_id = item["id"]
    try:
        output = await task(item=item)
    except Exception as exc:  # one broken item must not hide the rest of the run
        return ItemResult(node, item_id, "ERROR", f"{type(exc).__name__}: {exc}", {})
    if output.get("status") == "dead_lettered":
        return ItemResult(node, item_id, "ERROR", "node dead-lettered (LLM unreachable?)", output)
    if output.get("guard_reason") in _PARSE_FALLBACKS or output.get("reason") in _PARSE_FALLBACKS:
        # The node couldn't parse the LLM and returned its fallback verdict — which can look
        # like a pass (blocked, supported). Says nothing about prompt quality; never a PASS.
        return ItemResult(node, item_id, "ERROR", "LLM response unparseable — node used its fallback", output)
    failures = check_item(item["expected_output"], output)
    if failures:
        return ItemResult(node, item_id, "FAIL", "; ".join(failures), output)
    return ItemResult(node, item_id, "PASS", "", output)


async def run_async(cfg: NodeEvalConfig, llm: BaseChatModel | None = None, save: bool = True) -> int:
    """Run every configured node's dataset and return the process exit code.

    `llm` lets tests inject a mock; by default it's built from `deployment`.
    """
    datasets = [(spec.node, load_node_dataset(spec.dataset)) for spec in cfg.nodes]
    total = sum(len(items) for _, items in datasets)
    if total == 0:
        print(f"No items selected (nodes={[s.node for s in cfg.nodes]}) — failing closed.")
        return 1

    llm = llm or build_llm(NodeLLMConfig(model=cfg.deployment))
    print(f"\nNode eval — {len(datasets)} datasets, {total} items\n")
    results: list[ItemResult] = []
    for node, items in datasets:
        task = NODE_TASK_REGISTRY[node](llm)
        for item in items:
            r = await _run_item(node, task, item)
            print(f"  [{r.verdict:<5}] {node}/{r.item_id}" + (f"  — {r.detail}" if r.detail else ""))
            results.append(r)

    counts = {v: sum(1 for r in results if r.verdict == v) for v in ("PASS", "FAIL", "ERROR")}
    print(f"\n{'═' * 60}\n  " + "  ".join(f"{k}={n}" for k, n in counts.items()) + f"\n{'═' * 60}")
    if save:
        payload = {"config": cfg.model_dump(), "counts": counts, "results": [asdict(r) for r in results]}
        out = write_run("node_eval", payload)
        print(f"  Results written to {out}")
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


def run_from_config(path: Path) -> int:
    """evals/run.py entry point."""
    return asyncio.run(run_async(load_config(path)))
