"""Tests for evals/run_experiment.py's fail-fast dataset validation — Phase 5d.

Harness-integrity fix #3 (see PLAN.md): a misspelled expected_output field in a
hand-edited dataset YAML must raise at load time, not silently make an
evaluator a permanent no-op three layers downstream.
"""

import pytest

from evals.models import ExpectedOutput
from evals.run_experiment import _load_dataset, _validate_expected_outputs


class TestValidateExpectedOutputs:
    def test_valid_items_pass(self) -> None:
        items = [
            {
                "id": "case_1",
                "input": {"messages": [{"content": "hello"}]},
                "expected_output": {
                    "must_cover": [{"id": "kp1", "description": "x"}],
                    "must_include_terms": ["subprime"],
                    "must_not_contain": ["I cannot help with"],
                    "scoring_hints": ["hint"],
                },
            }
        ]
        _validate_expected_outputs("fixture.yaml", items)  # must not raise

    def test_misspelled_field_raises_with_file_and_item_context(self) -> None:
        items = [
            {
                "id": "case_1",
                "input": {"messages": [{"content": "hello"}]},
                "expected_output": {"must_coverr": [{"id": "kp1", "description": "x"}]},  # typo'd field
            }
        ]
        with pytest.raises(ValueError, match=r"fixture\.yaml.*case_1"):
            _validate_expected_outputs("fixture.yaml", items)

    def test_wrong_type_raises(self) -> None:
        items = [
            {
                "id": "case_1",
                "input": {"messages": [{"content": "hello"}]},
                "expected_output": {"must_include_terms": "subprime"},  # should be a list
            }
        ]
        with pytest.raises(ValueError):
            _validate_expected_outputs("fixture.yaml", items)

    def test_missing_input_key_raises_with_file_and_item_context(self) -> None:
        items = [{"id": "case_1", "expected_output": {}}]
        with pytest.raises(ValueError, match=r"fixture\.yaml.*case_1.*'input'"):
            _validate_expected_outputs("fixture.yaml", items)

    def test_missing_expected_output_key_raises_with_file_and_item_context(self) -> None:
        items = [{"id": "case_1", "input": {"messages": [{"content": "hello"}]}}]
        with pytest.raises(ValueError, match=r"fixture\.yaml.*case_1.*'expected_output'"):
            _validate_expected_outputs("fixture.yaml", items)


class TestLoadDataset:
    def test_sample_dataset_loads_and_validates(self) -> None:
        items = _load_dataset("sample.yaml")
        assert len(items) == 5
        for item in items:
            assert "input" in item
            assert len(ExpectedOutput.model_validate(item["expected_output"]).must_cover) >= 3, item["id"]
