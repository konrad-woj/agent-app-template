"""HTTP-driven experiment runner — Phase 5d (see PLAN.md).

Drives multi-turn conversations against a *live* agent-app server (POST
/v1/chat), reads the full checkpoint history directly from the same Postgres
database the server uses (not through a new API endpoint — see
`_read_history` below), and scores each run with the four evaluators in
evals/evaluators.py. Results are written to evals/.runs/<config-name>/<timestamp>.json
and, if Langfuse credentials are configured, also uploaded as scores.

Three harness-integrity fixes are baked in from this file's first version
(see PLAN.md Phase 5d for the audit that identified them as broadly reusable,
not agent-app-specific):

  1. A transport failure (httpx.HTTPError) during a run is caught inside the
     task function and returned as a scored "failed" item — never left to
     propagate and silently shrink the result count.
  2. An empty dataset (or, once Phase 5c ships file-based category filtering,
     an empty category selection) exits 1 immediately rather than reporting
     "0/0 passed" as a clean run.
  3. Every dataset item's expected_output is validated against
     evals.models.ExpectedOutput at load time, before any HTTP call is made —
     a typo'd field raises loudly instead of silently making an evaluator a
     permanent no-op.

Deviation from the original Phase 5d plan text, found while implementing:
    "variants with different backing models via AGENT_LLM_MODEL overrides" is
    not achievable against a single running server — agent-app compiles one
    LLM per node at startup (see app/graph/workflow.py:compile_graph); there is
    no per-request model override in POST /v1/chat (unlike node_llm_configs,
    which is a compile-time-only parameter — confirmed via
    tests/graph/nodes/test_node_config.py). A "variant" here is therefore a
    `base_url` pointing at a *different already-running* app instance
    (e.g. one started with AGENT_LLM_MODEL=A, another with =B), not a
    same-server model swap. See ExperimentVariant below.

Usage:
    uv run python evals/run_experiment.py evals/configs/exp_baseline.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.checkpoint.base import BaseCheckpointSaver
from logger import configure_logging, get_logger
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from evals.evaluators import (
    make_quality_score_evaluator,
    make_trace_assertion_evaluator,
    plan_approved_evaluator,
    turns_to_complete_evaluator,
)
from evals.models import ExpectedOutput

logger = get_logger(__name__)

_EVALS_DIR = Path(__file__).parent
_DATASETS_DIR = _EVALS_DIR / "datasets"
_CONFIGS_DIR = _EVALS_DIR / "configs"
_RUNS_DIR = _EVALS_DIR / ".runs"
_MAX_TURNS = 10  # safety cap; sample.yaml items are single-turn (one question + one approval)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class ExperimentVariant(BaseModel):
    name: str
    base_url: str = "http://localhost:8000"


class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset: str = "sample.yaml"  # resolved relative to evals/datasets/
    scoring_rubric: str = "scoring_rubric.yaml"  # resolved relative to evals/configs/
    variants: list[ExperimentVariant] = Field(default_factory=lambda: [ExperimentVariant(name="default")])
    judge_model: str | None = None  # None -> settings.llm_model (see evals/evaluators.py)
    db_uri: str | None = None  # None -> settings.db_uri (same Postgres the live server(s) use)
    max_concurrency: int = 3


# ---------------------------------------------------------------------------
# Dataset loading + fail-fast validation (harness-integrity fix #3)
# ---------------------------------------------------------------------------


def _validate_expected_outputs(dataset_file: str, items: list[dict]) -> None:
    """Validate every item has an `input` key and a schema-valid `expected_output`.

    Raises:
        ValueError: Naming the dataset file, item index, and item id/name for fast lookup —
            a missing key or misspelled field must be obvious, not a silent no-op or a raw
            KeyError three layers downstream (an HTTP call or a Langfuse upload).
    """
    for i, item in enumerate(items):
        label = item.get("id") or item.get("name") or f"index {i}"
        if "input" not in item:
            raise ValueError(f"{dataset_file}: item {i} ({label}) is missing required key 'input'")
        if "expected_output" not in item:
            raise ValueError(f"{dataset_file}: item {i} ({label}) is missing required key 'expected_output'")
        try:
            ExpectedOutput.model_validate(item["expected_output"])
        except ValidationError as exc:
            raise ValueError(f"{dataset_file}: item {i} ({label}) has an invalid expected_output: {exc}") from exc


def _load_dataset(dataset_file: str) -> list[dict]:
    path = _DATASETS_DIR / dataset_file
    data = yaml.safe_load(path.read_text())
    items: list[dict] = data.get("items", [])
    _validate_expected_outputs(str(path), items)
    return items


def _load_scoring_rubric(rubric_file: str) -> dict:
    path = _CONFIGS_DIR / rubric_file
    return yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------
# HTTP task — drives one dataset item through POST /v1/chat
# ---------------------------------------------------------------------------


async def _run_conversation(
    client: httpx.AsyncClient, thread_id: str, messages: list[str], approve_plan: bool
) -> tuple[dict, int, bool]:
    """Drive one conversation to completion. Returns (final_response, turns_taken, reached_interrupt).

    Only the first message triggers the planner interrupt in the current graph
    design (see app/graph/workflow.py) — subsequent dataset messages (none exist
    in evals/datasets/sample.yaml today, but the loop supports them) would each
    start a *fresh* pipeline run on the same thread_id, matching how a real
    multi-turn conversation behaves per app/routers.py's docstring.
    """
    last_response: dict = {}
    reached_interrupt = False
    turns = 0

    for message in messages[:_MAX_TURNS]:
        turns += 1
        resp = await client.post("/v1/chat", json={"thread_id": thread_id, "message": message})
        resp.raise_for_status()
        last_response = resp.json()

        if last_response.get("is_interrupted"):
            reached_interrupt = True
            resume_message = "approved" if approve_plan else "rejected"
            resume_resp = await client.post(
                "/v1/chat",
                json={"thread_id": thread_id, "message": resume_message, "approve": approve_plan},
            )
            resume_resp.raise_for_status()
            last_response = resume_resp.json()

    return last_response, turns, reached_interrupt


def make_task(base_url: str, checkpointer: BaseCheckpointSaver):
    """Return a task function closed over `base_url` and a shared `checkpointer` (one pool per variant).

    The HTTP conversation and the Postgres history read are isolated in separate
    try/except blocks: a transport failure never gets a history read attempted at
    all (nothing to read yet), but a history-read failure *after* a successful HTTP
    call must not discard the model's actual response and understate pass rates —
    it's reported as a run with an empty history/final_state instead of a full
    "task_error", which would otherwise mask a good answer behind a transient DB blip.
    """

    async def task(*, item: dict, **kwargs: Any) -> dict:
        _ = kwargs
        inp = item["input"]
        messages: list[str] = [m["content"] for m in inp["messages"]]
        approve_plan: bool = inp.get("approve_plan", True)
        thread_id = f"exp-{uuid.uuid4().hex[:12]}"

        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
                response, turns, reached_interrupt = await _run_conversation(client, thread_id, messages, approve_plan)
        except Exception as exc:
            # Harness-integrity fix #1: never let a per-item HTTP transport failure
            # propagate into the orchestration loop and silently shrink the result
            # count — return a scored failure instead. Broad `except Exception` is
            # deliberate here: this is a per-item isolation boundary in a batch job,
            # not application code — one bad item (server down, malformed response)
            # must not take the rest of the run down with it.
            logger.error("task.http_failed", thread_id=thread_id, error=str(exc), error_type=type(exc).__name__)
            return {
                "thread_id": thread_id,
                "status": "task_error",
                "final_answer": None,
                "question": messages[0] if messages else "",
                "approve_plan": approve_plan,
                "turns": 0,
                "reached_interrupt": False,
                "history": [],
                "final_state": {},
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            history, final_state = await _read_history(checkpointer, thread_id)
        except Exception as exc:
            # A DB blip here must not discard a perfectly good model response — the
            # HTTP-derived fields below are still reported; only the trace-dependent
            # fields (history, final_state, and therefore trace_assertions) are empty.
            logger.error("task.history_read_failed", thread_id=thread_id, error=str(exc), error_type=type(exc).__name__)
            history, final_state = [], {}

        return {
            "thread_id": thread_id,
            "status": response.get("status", "unknown"),
            "final_answer": response.get("final_answer"),
            "guard_reason": response.get("guard_reason"),
            "dead_letter": response.get("dead_letter"),
            "question": messages[0] if messages else "",
            "approve_plan": approve_plan,
            "turns": turns,
            "reached_interrupt": reached_interrupt,
            "history": history,
            "final_state": final_state,
        }

    return task


# ---------------------------------------------------------------------------
# Checkpoint history — read directly from Postgres, not a new API endpoint.
#
# CORRECTED (found by empirical test, not assumed — see PLAN.md Phase 5d):
# LangGraph derives StateSnapshot.next from the *reading* graph's own node/
# trigger topology (Pregel._prepare_state_snapshot -> prepare_next_tasks(...,
# processes=self.nodes, ...)), not from data stored in the checkpoint itself.
# compile_simple_graph's 2-node stub (planner -> writer, used by Phase 1's
# checkpointing/time-travel tests) can therefore never report "next" as any
# other node — a checkpoint written by the real graph with next=("input_guard",)
# reads back as next=() through the stub. Every check: checkpoint trace
# assertion (node_executed, graph_interrupted_at, all_nodes_executed) would
# have silently produced wrong results. Fixed by reading through
# app.graph.workflow.create_graph() instead — the same node/edge topology the
# live server actually compiled (create_graph() builds it with a lazily-
# constructed LLM and a no-op guard; neither is ever invoked here, since we
# only call aget_state_history(), never .ainvoke()).
# ---------------------------------------------------------------------------


async def _read_history(checkpointer: BaseCheckpointSaver, thread_id: str) -> tuple[list[dict], dict]:
    """Return (history, final_state): checkpoints newest-first + the newest checkpoint's values.

    `checkpointer` is shared across every item in a variant's run (see `_run_variant`)
    rather than opened fresh per item — one Postgres connection pool for the whole
    variant instead of one connection per dataset item.
    """
    from app.graph.workflow import create_graph

    graph = create_graph().compile(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = [
        {
            "step": (snapshot.metadata or {}).get("step", 0),
            "source": (snapshot.metadata or {}).get("source", ""),
            "next": snapshot.next,
            "values": snapshot.values,
        }
        async for snapshot in graph.aget_state_history(config)
    ]
    final_state = history[0]["values"] if history else {}
    return history, final_state


# ---------------------------------------------------------------------------
# Langfuse — optional. Scores are uploaded only if credentials are configured;
# the local run always completes and writes its JSON/log report regardless.
#
# LIMITATION: scores are uploaded as orphan scores keyed by our own locally
# generated thread_id (create_score(trace_id=thread_id, ...)) — there is no
# actual Langfuse trace with spans behind that ID, since app/routers.py's
# POST /v1/chat isn't instrumented with a Langfuse CallbackHandler and doesn't
# accept/propagate a trace-id header. Scores are visible in the Langfuse UI
# but not linked to per-node timing/token detail. Wiring real distributed
# tracing (CallbackHandler in node config["callbacks"], a trace-id request
# header like X-Langfuse-Trace-Id) is an app-level instrumentation change,
# not an eval-harness one — out of scope here; noted so it isn't mistaken for
# already having "traces" just because the original Phase 5d plan text said
# "uploads traces + scores."
# ---------------------------------------------------------------------------


def _langfuse_configured() -> bool:
    import os

    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _upload_scores(langfuse: Langfuse, thread_id: str, evaluations: list) -> None:
    for ev in evaluations:
        try:
            langfuse.create_score(
                trace_id=thread_id,
                name=ev.name,
                value=ev.value,
                data_type=ev.data_type or "NUMERIC",
                comment=ev.comment,
            )
        except Exception as exc:  # pragma: no cover - defensive; create_score itself rarely raises
            logger.warning("langfuse.score_upload_failed", thread_id=thread_id, name=ev.name, error=str(exc))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _run_variant(
    variant: ExperimentVariant,
    items: list[dict],
    evaluators: list,
    max_concurrency: int,
    langfuse: Langfuse | None,
    db_uri: str | None,
) -> dict:
    # One pooled checkpointer per variant, shared across every item's history read
    # below — matches app/main.py's own lifespan pattern (AsyncConnectionPool +
    # AsyncPostgresSaver(pool)) instead of each item opening and tearing down its
    # own single Postgres connection. Bounding the pool to max_concurrency matches
    # the same semaphore-limited number of items that can be mid-flight at once.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import AsyncConnectionPool

    async with AsyncConnectionPool(
        conninfo=db_uri or settings.db_uri,
        min_size=1,
        max_size=max_concurrency,
        kwargs={"autocommit": True, "row_factory": dict_row},
        connection_class=AsyncConnection[DictRow],
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        task_fn = make_task(variant.base_url, checkpointer)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(item: dict) -> dict:
            # The whole per-item pipeline (HTTP task + evaluators, including the
            # quality_score_evaluator's own LLM-judge call) is throttled together —
            # max_concurrency bounds concurrent *items*, not just the HTTP phase.
            # Evaluating outside the semaphore would let judge-LLM calls for every
            # item fire concurrently regardless of max_concurrency, defeating a
            # setting a caller lowers specifically to respect a rate-limited backend.
            async with semaphore:
                output = await task_fn(item=item)
                evaluations = []
                for evaluator in evaluators:
                    result = evaluator(output=output, expected_output=item.get("expected_output"))
                    if asyncio.iscoroutine(result):
                        result = await result
                    evaluations.append(result)
            if langfuse is not None:
                _upload_scores(langfuse, output["thread_id"], evaluations)
            return {
                "item_id": item.get("id"),
                "output": output,
                "evaluations": [{"name": e.name, "value": e.value, "comment": e.comment} for e in evaluations],
            }

        item_results = await asyncio.gather(*(_run_one(item) for item in items))

    summary: dict[str, list[float]] = {}
    for item_result in item_results:
        for ev in item_result["evaluations"]:
            summary.setdefault(ev["name"], []).append(ev["value"])
    aggregated = {name: {"mean": sum(vals) / len(vals), "n": len(vals)} for name, vals in summary.items()}

    return {"name": variant.name, "base_url": variant.base_url, "summary": aggregated, "item_results": item_results}


async def run_experiment(cfg: ExperimentConfig, cfg_file: str) -> int:
    """Run every variant against the dataset and write a report.

    Returns:
        0 on a clean run, 1 if the dataset resolved to zero items (harness-integrity
        fix #2 — fail closed rather than report "0/0 passed" as success).
    """
    items = _load_dataset(cfg.dataset)
    if not items:
        logger.error("run_experiment.empty_dataset", dataset=cfg.dataset)
        print(f"No items found in dataset {cfg.dataset!r} — failing closed rather than reporting a false pass.")
        return 1

    rubric = _load_scoring_rubric(cfg.scoring_rubric)
    evaluators = [
        make_quality_score_evaluator(rubric["quality_criteria"], cfg.judge_model),
        make_trace_assertion_evaluator(rubric["trace_assertions"]),
        turns_to_complete_evaluator,
        plan_approved_evaluator,
    ]

    langfuse: Langfuse | None = None
    if _langfuse_configured():
        langfuse = Langfuse()
        logger.info("run_experiment.langfuse_enabled")
    else:
        logger.info("run_experiment.langfuse_disabled", reason="LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set")

    logger.info(
        "run_experiment.start",
        experiment_name=cfg.experiment_name,
        dataset=cfg.dataset,
        item_count=len(items),
        variants=[v.name for v in cfg.variants],
    )

    variant_results = [
        await _run_variant(variant, items, evaluators, cfg.max_concurrency, langfuse, cfg.db_uri)
        for variant in cfg.variants
    ]

    if langfuse is not None:
        langfuse.flush()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    results_dir = _RUNS_DIR / Path(cfg_file).stem
    results_dir.mkdir(parents=True, exist_ok=True)

    run_data = {
        "experiment_file": cfg_file,
        "experiment_name": cfg.experiment_name,
        "dataset": cfg.dataset,
        "timestamp": timestamp,
        "langfuse_uploaded": langfuse is not None,
        "variants": variant_results,
    }
    json_path = results_dir / f"{timestamp}.json"
    json_path.write_text(json.dumps(run_data, indent=2, default=str))

    log_lines = [
        f"experiment_file: {cfg_file}",
        f"experiment_name: {cfg.experiment_name}",
        f"dataset: {cfg.dataset} ({len(items)} items)",
        f"timestamp: {timestamp}",
        f"langfuse_uploaded: {langfuse is not None}",
        "",
    ]
    for vr in variant_results:
        log_lines.append(f"  variant: {vr['name']} ({vr['base_url']})")
        for name, stats in vr["summary"].items():
            log_lines.append(f"    {name}: mean={stats['mean']:.3f} (n={stats['n']})")
    (results_dir / f"{timestamp}.log").write_text("\n".join(log_lines))

    logger.info("run_experiment.done", path=str(results_dir), json=str(json_path))
    print(f"Results saved -> {results_dir}/{timestamp}.[json|log]")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run an HTTP-driven eval experiment against agent-app.")
    parser.add_argument("config", help="Path to an experiment config YAML (e.g. evals/configs/exp_baseline.yaml)")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = ExperimentConfig.model_validate(raw)
    exit_code = asyncio.run(run_experiment(cfg, args.config))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
