import pytest

from rageval.evaluator import Evaluator
from rageval.metrics.structured import (
    ExactMatchMetric,
    NumericToleranceMetric,
    RefusalMetric,
    SQLEquivalenceMetric,
)
from rageval.types import QuestionType, RAGResponse, SQLResult, TestCase, TestSuite


def _case(**kwargs: object) -> TestCase:
    data: dict[str, object] = {
        "id": "case-1",
        "question": "Question?",
        "question_type": QuestionType.FACTUAL,
    }
    data.update(kwargs)
    return TestCase.model_validate(data)


def _response(
    answer: str = "Answer.",
    *,
    sql_result: SQLResult | None = None,
    refused: bool = False,
) -> RAGResponse:
    return RAGResponse(answer=answer, sql_result=sql_result, refused=refused)


class FakeSystem:
    async def answer(self, question: str) -> RAGResponse:  # noqa: ARG002
        return RAGResponse(answer="Luka Doncic led with 33.9 PPG.")


# ---------------------------------------------------------------------------
# ExactMatchMetric
# ---------------------------------------------------------------------------


def test_exact_match_exact() -> None:
    metric = ExactMatchMetric()
    result = metric(_case(expected_answer="Luka led."), _response("Luka led."))

    assert result.value == 1.0
    assert result.error is None


def test_exact_match_normalizes_case_and_whitespace() -> None:
    metric = ExactMatchMetric()
    case = _case(expected_answer="Luka Doncic led with 33.9 PPG.")
    response = _response("  luka   doncic LED with 33.9   ppg. ")

    result = metric(case, response)

    assert result.value == 1.0


def test_exact_match_mismatch() -> None:
    metric = ExactMatchMetric()
    result = metric(_case(expected_answer="Luka led."), _response("Tatum led."))

    assert result.value == 0.0
    assert result.error is None


def test_exact_match_missing_expected_answer() -> None:
    result = ExactMatchMetric()(_case(), _response("Anything."))

    assert result.value == 0.0
    assert result.error is not None
    assert "expected answer" in result.error.lower()


# ---------------------------------------------------------------------------
# NumericToleranceMetric
# ---------------------------------------------------------------------------


def test_numeric_tolerance_exact_match() -> None:
    case = _case(expected_numeric=33.9)
    result = NumericToleranceMetric()(case, _response("Luka averaged 33.9 points."))

    assert result.value == 1.0


def test_numeric_tolerance_within_tolerance() -> None:
    case = _case(expected_numeric=33.9, numeric_tolerance=0.2)
    result = NumericToleranceMetric()(case, _response("Luka averaged 33.8 points."))

    assert result.value == 1.0


def test_numeric_tolerance_outside_tolerance() -> None:
    case = _case(expected_numeric=33.9, numeric_tolerance=0.05)
    result = NumericToleranceMetric()(case, _response("Luka averaged 33.7 points."))

    assert result.value == 0.0
    assert result.details["delta"] == pytest.approx(0.2)


def test_numeric_tolerance_decimal_value() -> None:
    case = _case(expected_numeric=0.701, numeric_tolerance=0.001)
    result = NumericToleranceMetric()(case, _response("His true shooting was 0.701."))

    assert result.value == 1.0


def test_numeric_tolerance_percentage_converts_to_fraction() -> None:
    case = _case(expected_numeric=0.427, numeric_tolerance=0.001)
    result = NumericToleranceMetric()(case, _response("Curry shot 42.7%."))

    assert result.value == 1.0
    assert result.details["matched_numeric"] == pytest.approx(0.427)


def test_numeric_tolerance_percentage_requires_decimal_expected_value() -> None:
    case = _case(expected_numeric=42.7, numeric_tolerance=0.001)
    result = NumericToleranceMetric()(case, _response("Curry shot 42.7%."))

    assert result.value == 0.0
    assert result.details["matched_numeric"] == pytest.approx(0.427)


def test_numeric_tolerance_comma_separated_number() -> None:
    case = _case(expected_numeric=1234.0)
    result = NumericToleranceMetric()(case, _response("The total was 1,234 points."))

    assert result.value == 1.0


def test_numeric_tolerance_uses_best_candidate() -> None:
    case = _case(expected_numeric=33.9, numeric_tolerance=0.01)
    result = NumericToleranceMetric()(
        case,
        _response("In 2023-24, Luka led with 33.9 points per game."),
    )

    assert result.value == 1.0
    assert result.details["matched_numeric"] == pytest.approx(33.9)


def test_numeric_tolerance_missing_expected_numeric() -> None:
    result = NumericToleranceMetric()(_case(), _response("33.9"))

    assert result.value == 0.0
    assert result.error is not None
    assert "expected numeric" in result.error.lower()


def test_numeric_tolerance_answer_with_no_number() -> None:
    case = _case(expected_numeric=33.9)
    result = NumericToleranceMetric()(case, _response("Luka led the league."))

    assert result.value == 0.0
    assert result.error is not None
    assert "no numeric" in result.error.lower()


# ---------------------------------------------------------------------------
# SQLEquivalenceMetric
# ---------------------------------------------------------------------------


def test_sql_equivalence_exact_rows_match() -> None:
    rows = [{"player_name": "Luka Doncic", "points_per_game": 33.9}]
    case = _case(expected_sql_rows=rows)
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=rows))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 1.0


def test_sql_equivalence_order_insensitive_rows() -> None:
    expected = [{"player": "A"}, {"player": "B"}]
    actual = [{"player": "B"}, {"player": "A"}]
    case = _case(expected_sql_rows=expected)
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=actual))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 1.0


def test_sql_equivalence_row_mismatch() -> None:
    case = _case(expected_sql_rows=[{"player": "A"}])
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=[{"player": "B"}]))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 0.0


def test_sql_equivalence_key_mismatch() -> None:
    case = _case(expected_sql_rows=[{"player": "A"}])
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=[{"name": "A"}]))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 0.0


def test_sql_equivalence_is_type_strict() -> None:
    case = _case(expected_sql_rows=[{"points": 33}])
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=[{"points": "33"}]))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 0.0
    assert result.error is None


def test_sql_equivalence_missing_sql_result() -> None:
    case = _case(expected_sql_rows=[{"player": "A"}])
    result = SQLEquivalenceMetric()(case, _response())

    assert result.value == 0.0
    assert result.error is not None
    assert "sql result" in result.error.lower()


def test_sql_equivalence_sql_result_error() -> None:
    case = _case(expected_sql_rows=[{"player": "A"}])
    response = _response(sql_result=SQLResult(query="SELECT bad", rows=[], error="no column"))

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 0.0
    assert result.error is not None
    assert "no column" in result.error


def test_sql_equivalence_missing_expected_rows() -> None:
    response = _response(sql_result=SQLResult(query="SELECT ...", rows=[]))
    result = SQLEquivalenceMetric()(_case(), response)

    assert result.value == 0.0
    assert result.error is not None
    assert "expected sql rows" in result.error.lower()


def test_sql_equivalence_hybrid_allows_subset_contains_behavior() -> None:
    case = _case(
        question_type=QuestionType.HYBRID,
        expected_sql_rows=[{"player_name": "Nikola Jokic", "metric": "true_shooting_pct"}],
    )
    response = _response(
        sql_result=SQLResult(
            query="SELECT ...",
            rows=[
                {
                    "player_name": "Nikola Jokic",
                    "metric": "true_shooting_pct",
                    "value": 0.701,
                },
                {"player_name": "Other", "metric": "points_per_game", "value": 20.0},
            ],
        )
    )

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 1.0
    assert result.details["contains_match"] is True


def test_sql_equivalence_factual_requires_same_row_count() -> None:
    case = _case(expected_sql_rows=[{"player": "A"}])
    response = _response(
        sql_result=SQLResult(query="SELECT ...", rows=[{"player": "A"}, {"player": "B"}])
    )

    result = SQLEquivalenceMetric()(case, response)

    assert result.value == 0.0


# ---------------------------------------------------------------------------
# RefusalMetric
# ---------------------------------------------------------------------------


def test_refusal_should_refuse_and_did_refuse() -> None:
    result = RefusalMetric()(_case(should_refuse=True), _response(refused=True))

    assert result.value == 1.0
    assert result.details == {"expected_refused": True, "actual_refused": True}


def test_refusal_should_refuse_but_did_not() -> None:
    result = RefusalMetric()(_case(should_refuse=True), _response(refused=False))

    assert result.value == 0.0


def test_refusal_should_not_refuse_and_did_not() -> None:
    result = RefusalMetric()(_case(should_refuse=False), _response(refused=False))

    assert result.value == 1.0


def test_refusal_should_not_refuse_but_did() -> None:
    result = RefusalMetric()(_case(should_refuse=False), _response(refused=True))

    assert result.value == 0.0


# ---------------------------------------------------------------------------
# Evaluator compatibility
# ---------------------------------------------------------------------------


async def test_structured_metrics_work_with_evaluator() -> None:
    suite = TestSuite(
        name="structured-suite",
        cases=[
            _case(
                expected_answer="Luka Doncic led with 33.9 PPG.",
                expected_numeric=33.9,
                numeric_tolerance=0.01,
            )
        ],
    )

    result = await Evaluator(
        metrics=[ExactMatchMetric(), NumericToleranceMetric()]
    ).evaluate(FakeSystem(), suite)

    assert result.aggregate_scores == {
        "exact_match": 1.0,
        "numeric_tolerance": 1.0,
    }
