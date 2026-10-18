"""Tests for evals/run.py (dispatcher) and evals/node_eval.py (node_eval runner) — Phase 5e."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from evals import node_eval, run
from evals.node_eval import NodeEvalConfig, NodeSpec, check_item, load_node_dataset, run_async
from evals.node_tasks import NODE_TASK_REGISTRY


class TestResolve:
    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [("exp_baseline", "experiment"), ("guardrail_redteam", "guardrail_redteam"), ("node_eval", "node_eval")],
    )
    def test_bare_name_resolves_to_config_and_type(self, name: str, expected_type: str) -> None:
        path, config_type = run.resolve(name)
        assert path == run.CONFIGS_DIR / f"{name}.yaml"
        assert config_type == expected_type

    def test_name_with_yaml_suffix_and_explicit_path_resolve(self) -> None:
        assert run.resolve("node_eval.yaml")[1] == "node_eval"
        assert run.resolve(str(run.CONFIGS_DIR / "node_eval.yaml"))[1] == "node_eval"

    def test_unknown_name_lists_available_configs(self) -> None:
        with pytest.raises(run.ConfigError) as exc:
            run.resolve("does_not_exist")
        for name in ("exp_baseline", "guardrail_redteam", "node_eval"):
            assert name in str(exc.value)

    def test_config_without_runnable_type_is_rejected(self) -> None:
        with pytest.raises(run.ConfigError, match="type"):
            run.resolve("scoring_rubric")

    def test_available_configs_lists_only_runnable_ones(self) -> None:
        available = run.available_configs()
        assert available == {
            "exp_baseline": "experiment",
            "guardrail_redteam": "guardrail_redteam",
            "node_eval": "node_eval",
        }


class TestDispatch:
    @pytest.mark.parametrize(
        ("name", "module"),
        [
            ("exp_baseline", "evals.run_experiment"),
            ("guardrail_redteam", "evals.guardrail_redteam"),
            ("node_eval", "evals.node_eval"),
        ],
    )
    def test_dispatches_to_the_matching_runner(self, name: str, module: str) -> None:
        with patch(f"{module}.run_from_config", return_value=7) as runner:
            assert run.dispatch(name) == 7
        runner.assert_called_once_with(run.CONFIGS_DIR / f"{name}.yaml")

    def test_main_exits_2_with_a_clear_error_on_unknown_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["run.py", "nope"]), pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 2
        assert "no config named 'nope'" in capsys.readouterr().err


class TestBadConfigs:
    def test_broken_yaml_in_configs_dir_does_not_break_listing(self, tmp_path: Path) -> None:
        (tmp_path / "good.yaml").write_text("type: node_eval\n")
        (tmp_path / "scratch.yaml").write_text("type: [unclosed\n")
        with patch.object(run, "CONFIGS_DIR", tmp_path):
            assert run.available_configs() == {"good": "node_eval"}

    def test_invalid_yaml_config_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("type: [unclosed\n")
        with patch("sys.argv", ["run.py", str(bad)]), pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 2
        assert "cannot read config" in capsys.readouterr().err

    def test_config_failing_runner_validation_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("type: node_eval\nnodes:\n  - node: critic\n    dataset: node/x.yaml\n")
        with patch("sys.argv", ["run.py", str(bad)]), pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 2
        assert "unknown node" in capsys.readouterr().err

    def test_bare_name_prefers_configs_dir_over_a_local_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "node_eval").write_text("not a config")
        monkeypatch.chdir(tmp_path)
        assert run.resolve("node_eval") == (run.CONFIGS_DIR / "node_eval.yaml", "node_eval")

    def test_local_path_still_resolves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "mine.yaml").write_text("type: node_eval\n")
        monkeypatch.chdir(tmp_path)
        assert run.resolve("mine.yaml") == (Path("mine.yaml"), "node_eval")


class TestNodeEvalConfig:
    def test_shipped_config_covers_every_node_task(self) -> None:
        cfg = node_eval.load_config(run.CONFIGS_DIR / "node_eval.yaml")
        assert {s.node for s in cfg.nodes} == set(NODE_TASK_REGISTRY)

    def test_unknown_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown node"):
            NodeSpec(node="critic", dataset="node/x.yaml")

    def test_every_shipped_dataset_loads(self) -> None:
        cfg = node_eval.load_config(run.CONFIGS_DIR / "node_eval.yaml")
        for spec in cfg.nodes:
            assert load_node_dataset(spec.dataset), spec.dataset

    def test_unknown_expectation_key_fails_fast(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("items:\n  - id: x\n    input: {}\n    expected_output: {min_plan_step: 2}\n")
        with patch.object(node_eval, "_DATASETS_DIR", tmp_path), pytest.raises(ValueError, match="min_plan_step"):
            load_node_dataset("bad.yaml")


class TestCheckItem:
    def test_all_expectations_met(self) -> None:
        out = {"plan": ["a", "b"], "reached_interrupt": True}
        assert check_item({"min_plan_steps": 2, "must_reach_interrupt": True}, out) == []

    def test_reports_every_failed_expectation(self) -> None:
        failures = check_item({"min_draft_length": 10, "min_claims": 2}, {"draft_answer": "short", "claims": []})
        assert len(failures) == 2

    def test_has_verdict_requires_all_three_fields(self) -> None:
        assert check_item({"has_verdict": True}, {"supported": True, "confidence": None, "reason": "r"})


def _mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.metadata = None
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


class TestRunAsync:
    async def test_no_llm_nodes_pass_against_their_shipped_datasets(self) -> None:
        # output_guard and resume_guard make no LLM call, so their datasets are fully deterministic.
        cfg = NodeEvalConfig(
            nodes=[
                NodeSpec(node="output_guard", dataset="node/output_guard.yaml"),
                NodeSpec(node="resume_guard", dataset="node/resume_guard.yaml"),
            ]
        )
        assert await run_async(cfg, llm=_mock_llm("unused"), save=False) == 0

    async def test_unmet_expectation_fails_the_run(self) -> None:
        # An always-"safe" topic check can't block the off-topic input_guard items.
        cfg = NodeEvalConfig(nodes=[NodeSpec(node="input_guard", dataset="node/input_guard.yaml")])
        assert await run_async(cfg, llm=_mock_llm('{"verdict": "safe", "reason": "ok"}'), save=False) == 1

    async def test_dead_lettered_node_is_an_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from app.exceptions import LLMServiceUnavailableError

        llm = MagicMock()
        llm.metadata = {"max_retries": 0}
        llm.ainvoke = AsyncMock(side_effect=LLMServiceUnavailableError("down"))
        cfg = NodeEvalConfig(nodes=[NodeSpec(node="writer", dataset="node/writer.yaml")])
        assert await run_async(cfg, llm=llm, save=False) == 1
        assert "[ERROR] writer/" in capsys.readouterr().out

    @pytest.mark.parametrize("node", ["verifier", "input_guard", "planner"])
    async def test_unparseable_llm_never_passes(self, node: str, capsys: pytest.CaptureFixture[str]) -> None:
        # Garbage output makes these nodes fall back to verdicts that look valid (supported=True,
        # status=blocked); those must report as ERROR, not satisfy has_verdict / expected_status.
        cfg = NodeEvalConfig(nodes=[NodeSpec(node=node, dataset=f"node/{node}.yaml")])
        assert await run_async(cfg, llm=_mock_llm("not json at all"), save=False) == 1
        out = capsys.readouterr().out
        assert "[PASS ]" not in out
        assert "unparseable" in out

    async def test_empty_selection_fails_closed(self) -> None:
        assert await run_async(NodeEvalConfig(nodes=[]), llm=_mock_llm(""), save=False) == 1
