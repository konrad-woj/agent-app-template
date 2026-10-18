"""Single eval entrypoint — Phase 5e (see PLAN.md).

    uv run python evals/run.py <config-name | path/to/config.yaml>

A bare name resolves to evals/configs/<name>.yaml. The config's `type:` field
picks the runner:

    experiment         evals/run_experiment.py   HTTP-driven end-to-end experiment
    guardrail_redteam  evals/guardrail_redteam.py categorized guard attack suite
    node_eval          evals/node_eval.py         per-node quality checks vs a real LLM

Every runner exposes `load_config(path)` and `run_from_config(path) -> int` and
is imported only when selected, so a config for one runner never needs the
others' dependencies. The config is validated before the run starts.
The exit code is the runner's (0 clean, 1 failures or an empty selection);
2 means the config itself couldn't be resolved.

This is a CLI convention, not an eval-methodology choice. evals/smoke_test.py
has no config and stays a standalone script; runner-specific flags (e.g.
guardrail_redteam's --include-multiturn) stay on the individual scripts.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

CONFIGS_DIR = Path(__file__).parent / "configs"

# type → module exposing run_from_config(path) -> int
RUNNERS: dict[str, str] = {
    "experiment": "evals.run_experiment",
    "guardrail_redteam": "evals.guardrail_redteam",
    "node_eval": "evals.node_eval",
}


class ConfigError(Exception):
    """The config couldn't be resolved or has no runnable `type`."""


def _read_type(path: Path) -> str | None:
    """Return the config's `type`, or None if it has none.

    Raises:
        ConfigError: the file can't be read or isn't valid YAML.
    """
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: cannot read config: {exc}") from exc
    return data.get("type") if isinstance(data, dict) else None


def available_configs() -> dict[str, str]:
    """{name: type} for every config in evals/configs/ that a runner can take.

    Unreadable files are skipped: one broken scratch file must not break the
    help text or the listing in every other error message.
    """
    found = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            config_type = _read_type(path)
        except ConfigError:
            continue
        if config_type in RUNNERS:
            found[path.stem] = config_type
    return found


def _available_text() -> str:
    return ", ".join(f"{name} ({t})" for name, t in available_configs().items()) or "none"


def resolve(arg: str) -> tuple[Path, str]:
    """Return (config path, runner type) for a bare name or a path.

    A bare name (no directory part) is looked up in evals/configs/ first, so a
    stray file of the same name in the working directory can't shadow it.

    Raises:
        ConfigError: unknown name, missing or unreadable file, or a missing/unknown `type`.
    """
    named = CONFIGS_DIR / f"{arg.removesuffix('.yaml')}.yaml"
    if Path(arg).parent == Path(".") and named.is_file():
        path = named
    elif Path(arg).is_file():
        path = Path(arg)
    else:
        raise ConfigError(f"no config named {arg!r}. Available: {_available_text()}")
    config_type = _read_type(path)
    if config_type not in RUNNERS:
        raise ConfigError(
            f"{path}: `type` is {config_type!r}, expected one of {sorted(RUNNERS)}. Available: {_available_text()}"
        )
    return path, config_type


def dispatch(arg: str) -> int:
    """Resolve, validate and run a config; return the runner's exit code.

    Raises:
        ConfigError: the config can't be resolved or fails its runner's validation.
    """
    path, config_type = resolve(arg)
    runner = importlib.import_module(RUNNERS[config_type])
    try:
        runner.load_config(path)
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid {config_type} config:\n{exc}") from exc
    return runner.run_from_config(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an eval by config name (evals/configs/<name>.yaml) or path.",
        epilog=f"Available configs: {_available_text()}",
    )
    parser.add_argument("config", help="Config name (e.g. exp_baseline) or path to a YAML config")
    args = parser.parse_args()
    try:
        sys.exit(dispatch(args.config))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
