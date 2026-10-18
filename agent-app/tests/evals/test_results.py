"""Tests for evals/_results.py — shared run-file naming and writing."""

import json
from pathlib import Path

import pytest

from evals import _results


class TestWriteRun:
    def test_writes_json_under_subdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_results, "RUNS_DIR", tmp_path)
        out = _results.write_run("node_eval", {"counts": {"PASS": 1}})
        assert out.parent == tmp_path / "node_eval"
        assert json.loads(out.read_text()) == {"counts": {"PASS": 1}}

    def test_runs_in_the_same_second_do_not_collide(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_results, "RUNS_DIR", tmp_path)
        paths = {_results.write_run("redteam", {"n": i}) for i in range(5)}
        assert len(paths) == 5

    def test_explicit_stamp_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_results, "RUNS_DIR", tmp_path)
        assert _results.write_run("exp", {}, stamp="fixed").name == "fixed.json"
