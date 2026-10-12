"""Node-level eval runner — Phase 5e (see PLAN.md).

Runs the Phase 5a node tasks (evals/node_tasks.py) against a real LLM, one
dataset per node (evals/datasets/node/*.yaml), and checks each item's
`expected_output`. tests/evals/test_node_tasks.py covers the same tasks with a
mocked LLM; this is the prompt/quality regression run.

Each item gets PASS, FAIL (an expectation not met), or ERROR (the task raised,
or the node dead-lettered — usually the LLM is unreachable). Exits 1 on any
FAIL or ERROR, or if the config selects zero items.

GLiGuard stays mocked permissively here (see evals/node_tasks.py): this tier
measures the LLM-driven decision each node makes. Guard behaviour is the
guardrail_redteam suite's job.

Usage:
    uv run python evals/run.py node_eval
"""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, field_validator

from app.graph.nodes._llm_invoke import NodeLLMConfig, build_llm
from evals.node_tasks import NODE_TASK_REGISTRY

_EVALS_DIR = Path(__file__).parent
_DATASETS_DIR = _EVALS_DIR / "datasets"
_RUNS_DIR = _EVALS_DIR / ".runs" / "node_eval"


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
    nodes: list[NodeSpec]
    deployment: str | None = None  # LiteLLM model id; None → AGENT_LLM_MODEL


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
    return None if got is want else f"has_verdict={got}, expected {want}: {out}"


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
    """Load one node dataset, failing fast on a missing key or an unknown expectation."""
    path = _DATASETS_DIR / dataset
    items: list[dict] = (yaml.safe_load(path.read_text()) or {}).get("items") or []
    for i, item in enumerate(items):
        label = item.get("id", f"index {i}")
        if "input" not in item or "expected_output" not in item:
            raise ValueError(f"{dataset}: item {label} needs both 'input' and 'expected_output'")
        unknown = set(item["expected_output"]) - CHECKS.keys()
        if unknown:
            raise ValueError(f"{dataset}: item {label} has unknown expected_output keys {sorted(unknown)}")
    return items


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemResult:
    node: str
    item_id: str
    verdict: Literal["PASS", "FAIL", "ERROR"]
    detail: str
    output: dict[str, Any]


async def _run_item(node: str, task: Callable, item: dict) -> ItemResult:
    item_id = item.get("id", "?")
    try:
        output = await task(item=item)
    except Exception as exc:  # one broken item must not hide the rest of the run
        return ItemResult(node, item_id, "ERROR", f"{type(exc).__name__}: {exc}", {})
    if output.get("status") == "dead_lettered":
        return ItemResult(node, item_id, "ERROR", "node dead-lettered (LLM unreachable?)", output)
    failures = check_item(item["expected_output"], output)
    if failures:
        return ItemResult(node, item_id, "FAIL", "; ".join(failures), output)
    return ItemResult(node, item_id, "PASS", "", output)


async def run_async(cfg: NodeEvalConfig, llm: BaseChatModel | None = None, save: bool = True) -> int:
    """Run every configured node's dataset and return the process exit code.

    `llm` lets tests inject a mock; by default it's built from `deployment`.
    """
    datasets = {spec.node: load_node_dataset(spec.dataset) for spec in cfg.nodes}
    total = sum(len(items) for items in datasets.values())
    if total == 0:
        print(f"No items selected (nodes={[s.node for s in cfg.nodes]}) — failing closed.")
        return 1

    llm = llm or build_llm(NodeLLMConfig(model=cfg.deployment))
    print(f"\nNode eval — {len(datasets)} nodes, {total} items\n")
    results: list[ItemResult] = []
    for node, items in datasets.items():
        task = NODE_TASK_REGISTRY[node](llm)
        for item in items:
            r = await _run_item(node, task, item)
            print(f"  [{r.verdict:<5}] {node}/{r.item_id}" + (f"  — {r.detail}" if r.detail else ""))
            results.append(r)

    counts = {v: sum(1 for r in results if r.verdict == v) for v in ("PASS", "FAIL", "ERROR")}
    print(f"\n{'═' * 60}\n  " + "  ".join(f"{k}={n}" for k, n in counts.items()) + f"\n{'═' * 60}")
    if save:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        out = _RUNS_DIR / f"{timestamp}.json"
        payload = {"config": cfg.model_dump(), "counts": counts, "results": [asdict(r) for r in results]}
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"  Results written to {out}")
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


def run_from_config(path: Path) -> int:
    """evals/run.py entry point."""
    return asyncio.run(run_async(load_config(path)))
