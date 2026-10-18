"""Where and how eval runners save their results: evals/.runs/<subdir>/<stamp>.json.

Shared by run_experiment.py, guardrail_redteam.py and node_eval.py so every
runner names its files the same way. The stamp is UTC time plus a short random
suffix, so two runs finishing in the same second (e.g. a CI matrix on one
runner) don't overwrite each other.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).parent / ".runs"


def run_stamp() -> str:
    """Return a sortable, collision-safe run id, e.g. 20261018T091500Z-3f9a1c."""
    return f"{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def write_run(subdir: str, payload: dict[str, Any], stamp: str | None = None) -> Path:
    """Write `payload` as evals/.runs/<subdir>/<stamp>.json and return the path."""
    out_dir = RUNS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp or run_stamp()}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
