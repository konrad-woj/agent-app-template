"""Upload a local dataset YAML to Langfuse — Phase 5d (see PLAN.md).

Purely optional: evals/run_experiment.py reads datasets directly from
evals/datasets/*.yaml and does not require this script to have been run.
Use this only if you want the dataset (and, over time, per-item score history)
visible in the Langfuse UI. Skips (with a warning, exit 0) rather than
crashing when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't configured —
Langfuse is an optional integration throughout this repo (see README.md).

Usage:
    uv run python evals/create_dataset.py evals/datasets/sample.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from langfuse import Langfuse
from logger import configure_logging, get_logger

from evals.run_experiment import _validate_expected_outputs  # reuse — same fail-fast check

logger = get_logger(__name__)


def sync_dataset(langfuse: Langfuse, dataset_file: Path) -> str:
    """Upload dataset YAML to Langfuse (idempotent) and return the dataset name."""
    data = yaml.safe_load(dataset_file.read_text())
    items: list[dict] = data.get("items", [])
    _validate_expected_outputs(str(dataset_file), items)

    dataset_name = data.get("dataset_name") or dataset_file.stem
    langfuse.create_dataset(name=dataset_name, description=data.get("description", ""))
    for item in items:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=item.get("id"),
            input=item["input"],
            expected_output=item["expected_output"],
        )
    return dataset_name


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Upload a dataset YAML to Langfuse.")
    parser.add_argument("dataset_file", help="Path to a dataset YAML (e.g. evals/datasets/sample.yaml)")
    args = parser.parse_args()

    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set — skipping (this script is optional).")
        sys.exit(0)

    langfuse = Langfuse()
    dataset_name = sync_dataset(langfuse, Path(args.dataset_file))
    logger.info("create_dataset.done", dataset_name=dataset_name)
    print(f"Uploaded dataset {dataset_name!r} to Langfuse.")


if __name__ == "__main__":
    main()
