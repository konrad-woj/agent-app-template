"""Tests for the shared pass policy in _rubric_models."""

from pydantic import BaseModel

from app.graph.nodes.subgraphs._rubric_models import CriterionVerdict, HolisticCriterionVerdict, apply_pass_policy


class _Verdict(BaseModel):
    detail: CriterionVerdict
    overall: HolisticCriterionVerdict
    summary: str


def _verdict(detail: int, overall: int, low: tuple[str, ...] = ()) -> _Verdict:
    def v(cls: type, name: str, score: int):
        return cls(reasoning="r", score=score, confidence="low" if name in low else "high", reason=name)

    return _Verdict(
        detail=v(CriterionVerdict, "detail", detail),
        overall=v(HolisticCriterionVerdict, "overall", overall),
        summary="",
    )


class TestApplyPassPolicy:
    def test_passes_when_minimums_and_total_met(self) -> None:
        result = apply_pass_policy(_verdict(3, 1), {"detail": 3, "overall": 1}, minimum_total=4)
        assert (result.passed, result.total, result.max_total, result.below_minimum) == (True, 4, 6, [])

    def test_fails_on_a_criterion_below_minimum_even_with_enough_total(self) -> None:
        result = apply_pass_policy(_verdict(4, 0), {"overall": 1}, minimum_total=4)
        assert result.passed is False
        assert result.below_minimum == ["overall"]

    def test_fails_on_total_alone(self) -> None:
        result = apply_pass_policy(_verdict(2, 1), {}, minimum_total=4)
        assert result.passed is False
        assert result.below_minimum == []

    def test_reports_low_confidence_criteria(self) -> None:
        result = apply_pass_policy(_verdict(4, 2, low=("overall",)), {}, minimum_total=0)
        assert result.low_confidence == ["overall"]
