from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rageval.types import (
    CaseResult,
    Document,
    EvaluationResult,
    MetricResult,
    QuestionType,
    RAGResponse,
    SQLResult,
    TestCase,
    TestSuite,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── QuestionType ──────────────────────────────────────────────────────────────


def test_question_type_values() -> None:
    assert QuestionType.FACTUAL.value == "factual"
    assert QuestionType.ANALYTICAL.value == "analytical"
    assert QuestionType.HYBRID.value == "hybrid"
    assert QuestionType.UNANSWERABLE.value == "unanswerable"


def test_question_type_from_string() -> None:
    assert QuestionType("factual") is QuestionType.FACTUAL
    assert QuestionType("hybrid") is QuestionType.HYBRID


def test_question_type_invalid() -> None:
    with pytest.raises(ValueError):
        QuestionType("unknown")


# ── Document ──────────────────────────────────────────────────────────────────


def test_document_minimal() -> None:
    doc = Document(id="doc-1", content="some text")
    assert doc.id == "doc-1"
    assert doc.content == "some text"
    assert doc.metadata == {}


def test_document_with_metadata() -> None:
    doc = Document(id="doc-1", content="text", metadata={"source": "ctg", "page": 2})
    assert doc.metadata["source"] == "ctg"
    assert doc.metadata["page"] == 2


def test_document_missing_content() -> None:
    with pytest.raises(ValidationError):
        Document(id="doc-1")  # type: ignore[call-arg]


def test_document_missing_id() -> None:
    with pytest.raises(ValidationError):
        Document(content="text")  # type: ignore[call-arg]


# ── SQLResult ─────────────────────────────────────────────────────────────────


def test_sql_result_basic() -> None:
    result = SQLResult(query="SELECT 1", rows=[{"val": 1}])
    assert result.error is None
    assert result.rows[0]["val"] == 1


def test_sql_result_empty_rows() -> None:
    result = SQLResult(query="SELECT 1 WHERE FALSE", rows=[])
    assert result.rows == []


def test_sql_result_with_error() -> None:
    result = SQLResult(query="BAD SQL", rows=[], error="syntax error near BAD")
    assert result.error == "syntax error near BAD"


def test_sql_result_missing_query() -> None:
    with pytest.raises(ValidationError):
        SQLResult(rows=[])  # type: ignore[call-arg]


# ── TestCase ──────────────────────────────────────────────────────────────────


def test_test_case_minimal() -> None:
    case = TestCase(id="t1", question="Q?", question_type=QuestionType.FACTUAL)
    assert case.relevant_doc_ids == []
    assert case.should_refuse is False
    assert case.numeric_tolerance == 0.01
    assert case.expected_sql_rows is None
    assert case.live_expected_sql_rows is None
    assert case.expected_numeric is None
    assert case.expected_answer is None
    assert case.metadata == {}


def test_test_case_factual_full() -> None:
    case = TestCase(
        id="factual-001",
        question="Who led in PPG?",
        question_type=QuestionType.FACTUAL,
        expected_sql_rows=[{"player_name": "Luka Dončić", "points_per_game": 33.9}],
        live_expected_sql_rows=[
            {"full_name": "Joel Embiid", "points_per_game": 34.7}
        ],
        expected_numeric=33.9,
        numeric_tolerance=0.05,
        expected_answer="Luka Dončić led with 33.9 PPG.",
    )
    assert case.expected_numeric == 33.9
    assert case.expected_sql_rows is not None
    assert case.expected_sql_rows[0]["player_name"] == "Luka Dončić"
    assert case.live_expected_sql_rows is not None
    assert case.live_expected_sql_rows[0]["full_name"] == "Joel Embiid"


def test_test_case_live_expected_sql_rows_optional() -> None:
    case = TestCase(id="t1", question="Q?", question_type=QuestionType.FACTUAL)

    assert case.live_expected_sql_rows is None


def test_test_case_live_expected_sql_rows_loads_from_dict() -> None:
    case = TestCase.model_validate(
        {
            "id": "factual-live",
            "question": "Who led in PPG?",
            "question_type": "factual",
            "expected_sql_rows": [{"player_name": "Luka Dončić"}],
            "live_expected_sql_rows": [
                {"full_name": "Joel Embiid", "points_per_game": 34.7}
            ],
        }
    )

    assert case.expected_sql_rows == [{"player_name": "Luka Dončić"}]
    assert case.live_expected_sql_rows == [
        {"full_name": "Joel Embiid", "points_per_game": 34.7}
    ]


def test_test_case_analytical() -> None:
    case = TestCase(
        id="analytical-001",
        question="What are the four factors?",
        question_type=QuestionType.ANALYTICAL,
        relevant_doc_ids=["ctg-four-factors#0", "ctg-four-factors#1"],
    )
    assert len(case.relevant_doc_ids) == 2
    assert case.expected_sql_rows is None


def test_test_case_adversarial() -> None:
    case = TestCase(
        id="adv-001",
        question="Who wins MVP in 2030?",
        question_type=QuestionType.UNANSWERABLE,
        should_refuse=True,
    )
    assert case.should_refuse is True
    assert case.relevant_doc_ids == []


def test_test_case_hybrid() -> None:
    case = TestCase(
        id="hybrid-001",
        question="How does Jokić compare analytically and statistically?",
        question_type=QuestionType.HYBRID,
        expected_sql_rows=[{"player_name": "Nikola Jokić", "metric": "ts_pct"}],
        relevant_doc_ids=["thinking-basketball-jokic#0"],
        metadata={"category": "hybrid", "difficulty": "hard"},
    )
    assert case.question_type == QuestionType.HYBRID
    assert case.metadata["difficulty"] == "hard"


def test_test_case_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        TestCase(id="t1", question="Q?")  # type: ignore[call-arg]


def test_test_case_invalid_question_type() -> None:
    with pytest.raises(ValidationError):
        TestCase(id="t1", question="Q?", question_type="not_valid")  # type: ignore[arg-type]


def test_test_case_question_type_as_string() -> None:
    # Pydantic should coerce the string to the enum
    case = TestCase(id="t1", question="Q?", question_type="analytical")  # type: ignore[arg-type]
    assert case.question_type is QuestionType.ANALYTICAL


# ── TestSuite ─────────────────────────────────────────────────────────────────


def test_test_suite_basic() -> None:
    case = TestCase(id="t1", question="Q?", question_type=QuestionType.FACTUAL)
    suite = TestSuite(name="my-suite", cases=[case])
    assert suite.name == "my-suite"
    assert suite.description == ""
    assert len(suite.cases) == 1


def test_test_suite_with_description() -> None:
    suite = TestSuite(name="s", description="A test suite.", cases=[])
    assert suite.description == "A test suite."


def test_test_suite_empty_cases() -> None:
    suite = TestSuite(name="empty", cases=[])
    assert suite.cases == []


def test_test_suite_missing_name() -> None:
    with pytest.raises(ValidationError):
        TestSuite(cases=[])  # type: ignore[call-arg]


def test_test_suite_missing_cases() -> None:
    with pytest.raises(ValidationError):
        TestSuite(name="s")  # type: ignore[call-arg]


# ── TestSuite.from_yaml ───────────────────────────────────────────────────────


def test_from_yaml_fixture() -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    assert suite.name == "fixture-suite"
    assert len(suite.cases) == 5


def test_from_yaml_case_types(tmp_path: Path) -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    types = {c.question_type for c in suite.cases}
    assert QuestionType.FACTUAL in types
    assert QuestionType.ANALYTICAL in types
    assert QuestionType.HYBRID in types
    assert QuestionType.UNANSWERABLE in types


def test_from_yaml_factual_fields() -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    factual = next(c for c in suite.cases if c.id == "factual-001")
    assert factual.expected_sql_rows is not None
    assert factual.expected_sql_rows[0]["player_name"] == "Luka Dončić"
    assert factual.live_expected_sql_rows is None
    assert factual.expected_answer is not None


def test_from_yaml_live_expected_sql_rows(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
name: live-suite
cases:
  - id: factual-live
    question: Who led in PPG?
    question_type: factual
    expected_sql_rows:
      - player_name: Luka Dončić
        points_per_game: 33.9
    live_expected_sql_rows:
      - full_name: Joel Embiid
        points_per_game: 34.7
""",
        encoding="utf-8",
    )

    suite = TestSuite.from_yaml(str(path))

    case = suite.cases[0]
    assert case.expected_sql_rows == [
        {"player_name": "Luka Dončić", "points_per_game": 33.9}
    ]
    assert case.live_expected_sql_rows == [
        {"full_name": "Joel Embiid", "points_per_game": 34.7}
    ]


def test_from_yaml_numeric_tolerance() -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    case = next(c for c in suite.cases if c.id == "factual-002")
    assert case.expected_numeric == pytest.approx(0.701)
    assert case.numeric_tolerance == pytest.approx(0.005)


def test_from_yaml_relevant_doc_ids() -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    case = next(c for c in suite.cases if c.id == "analytical-001")
    assert "ctg-four-factors#0" in case.relevant_doc_ids
    assert len(case.relevant_doc_ids) == 3


def test_from_yaml_adversarial_should_refuse() -> None:
    suite = TestSuite.from_yaml(str(FIXTURES / "valid_suite.yaml"))
    case = next(c for c in suite.cases if c.id == "adversarial-001")
    assert case.should_refuse is True
    assert case.question_type == QuestionType.UNANSWERABLE


def test_from_yaml_examples_file() -> None:
    suite = TestSuite.from_yaml("examples/nba_test_suite.yaml")
    assert suite.name == "nba-hybrid-demo"
    assert len(suite.cases) > 0


def test_from_yaml_inline_valid(tmp_path: Path) -> None:
    content = """\
name: inline-suite
description: Written inline.
cases:
  - id: f001
    question: Who led in PPG?
    question_type: factual
    expected_answer: Luka.
  - id: a001
    question: What are the four factors?
    question_type: analytical
    relevant_doc_ids:
      - ctg-four-factors#0
"""
    f = tmp_path / "suite.yaml"
    f.write_text(content)
    suite = TestSuite.from_yaml(str(f))
    assert suite.name == "inline-suite"
    assert suite.cases[0].id == "f001"
    assert suite.cases[1].relevant_doc_ids == ["ctg-four-factors#0"]


def test_from_yaml_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        TestSuite.from_yaml("/nonexistent/path/suite.yaml")


def test_from_yaml_invalid_yaml(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("name: test\ncases: [\nnot: valid: yaml: {{{{")
    with pytest.raises(ValueError, match="Invalid YAML"):
        TestSuite.from_yaml(str(f))


def test_from_yaml_top_level_not_mapping(tmp_path: Path) -> None:
    f = tmp_path / "list.yaml"
    f.write_text("- item one\n- item two\n")
    with pytest.raises(ValueError, match="mapping"):
        TestSuite.from_yaml(str(f))


def test_from_yaml_missing_required_name(tmp_path: Path) -> None:
    f = tmp_path / "no_name.yaml"
    f.write_text("cases:\n  - id: t1\n    question: Q?\n    question_type: factual\n")
    with pytest.raises(ValueError, match="Invalid test suite structure"):
        TestSuite.from_yaml(str(f))


def test_from_yaml_missing_question(tmp_path: Path) -> None:
    f = tmp_path / "no_question.yaml"
    f.write_text("name: s\ncases:\n  - id: t1\n    question_type: factual\n")
    with pytest.raises(ValueError, match="Invalid test suite structure"):
        TestSuite.from_yaml(str(f))


def test_from_yaml_invalid_enum_value(tmp_path: Path) -> None:
    f = tmp_path / "bad_enum.yaml"
    f.write_text(
        "name: s\ncases:\n  - id: t1\n    question: Q?\n    question_type: WRONG\n"
    )
    with pytest.raises(ValueError, match="Invalid test suite structure"):
        TestSuite.from_yaml(str(f))


# ── RAGResponse ───────────────────────────────────────────────────────────────


def test_rag_response_minimal() -> None:
    resp = RAGResponse(answer="The answer.")
    assert resp.retrieved_docs == []
    assert resp.sql_result is None
    assert resp.routing_decision is None
    assert resp.latency_ms is None
    assert resp.cost_usd is None
    assert resp.refused is False


def test_rag_response_full() -> None:
    doc = Document(id="d1", content="chunk text")
    sql = SQLResult(query="SELECT 1", rows=[{"val": 42}])
    resp = RAGResponse(
        answer="An answer.",
        retrieved_docs=[doc],
        sql_result=sql,
        routing_decision=QuestionType.HYBRID,
        latency_ms=312.5,
        cost_usd=0.0023,
        refused=False,
    )
    assert len(resp.retrieved_docs) == 1
    assert resp.sql_result is not None
    assert resp.sql_result.rows[0]["val"] == 42
    assert resp.routing_decision == QuestionType.HYBRID
    assert resp.latency_ms == pytest.approx(312.5)


def test_rag_response_refused() -> None:
    resp = RAGResponse(answer="I can't answer that.", refused=True)
    assert resp.refused is True


def test_rag_response_missing_answer() -> None:
    with pytest.raises(ValidationError):
        RAGResponse()  # type: ignore[call-arg]


# ── MetricResult ──────────────────────────────────────────────────────────────


def test_metric_result_basic() -> None:
    mr = MetricResult(metric_name="precision_at_5", case_id="t1", value=0.8)
    assert mr.details == {}
    assert mr.error is None


def test_metric_result_with_details() -> None:
    mr = MetricResult(
        metric_name="ndcg_at_10",
        case_id="t1",
        value=0.65,
        details={"k": 10, "relevant_count": 3},
    )
    assert mr.details["k"] == 10


def test_metric_result_with_error() -> None:
    mr = MetricResult(
        metric_name="ndcg_at_10",
        case_id="t1",
        value=0.0,
        error="no relevant docs defined",
    )
    assert mr.error == "no relevant docs defined"
    assert mr.value == 0.0


def test_metric_result_zero_value() -> None:
    mr = MetricResult(metric_name="precision_at_5", case_id="t1", value=0.0)
    assert mr.value == 0.0


# ── CaseResult ────────────────────────────────────────────────────────────────


def test_case_result_basic() -> None:
    resp = RAGResponse(answer="A")
    mr = MetricResult(metric_name="precision_at_5", case_id="c1", value=1.0)
    cr = CaseResult(case_id="c1", question="Q?", response=resp, metric_results=[mr])
    assert cr.case_id == "c1"
    assert cr.question == "Q?"
    assert len(cr.metric_results) == 1


def test_case_result_empty_metrics() -> None:
    resp = RAGResponse(answer="A")
    cr = CaseResult(case_id="c1", question="Q?", response=resp, metric_results=[])
    assert cr.metric_results == []


# ── EvaluationResult ──────────────────────────────────────────────────────────


def _make_case_result(case_id: str = "c1") -> CaseResult:
    resp = RAGResponse(answer="A")
    mr = MetricResult(metric_name="precision_at_5", case_id=case_id, value=1.0)
    return CaseResult(case_id=case_id, question="Q?", response=resp, metric_results=[mr])


def test_evaluation_result_basic() -> None:
    now = datetime.now(tz=UTC)
    er = EvaluationResult(
        suite_name="my-suite",
        system_name="demo-v1",
        run_at=now,
        case_results=[_make_case_result()],
        aggregate_scores={"precision_at_5": 1.0},
        total_cost_usd=0.012,
        total_duration_seconds=8.4,
    )
    assert er.suite_name == "my-suite"
    assert er.errors == []
    assert er.aggregate_scores["precision_at_5"] == 1.0


def test_evaluation_result_with_errors() -> None:
    now = datetime.now(tz=UTC)
    er = EvaluationResult(
        suite_name="s",
        system_name="sys",
        run_at=now,
        case_results=[],
        aggregate_scores={},
        total_cost_usd=0.0,
        total_duration_seconds=0.0,
        errors=["case t1 timed out", "case t2 raised ValueError"],
    )
    assert len(er.errors) == 2


def test_evaluation_result_multiple_metrics() -> None:
    now = datetime.now(tz=UTC)
    er = EvaluationResult(
        suite_name="s",
        system_name="sys",
        run_at=now,
        case_results=[_make_case_result("c1"), _make_case_result("c2")],
        aggregate_scores={"precision_at_5": 0.8, "ndcg_at_10": 0.72, "mrr": 0.9},
        total_cost_usd=0.05,
        total_duration_seconds=22.1,
    )
    assert len(er.case_results) == 2
    assert len(er.aggregate_scores) == 3


# ── Roundtrip serialization ───────────────────────────────────────────────────


def test_test_case_roundtrip() -> None:
    case = TestCase(
        id="t1",
        question="Q?",
        question_type=QuestionType.ANALYTICAL,
        relevant_doc_ids=["doc#0", "doc#1"],
        expected_answer="The answer.",
        metadata={"tag": "roundtrip"},
    )
    restored = TestCase.model_validate(case.model_dump())
    assert restored == case
    assert restored.question_type is QuestionType.ANALYTICAL


def test_test_suite_roundtrip() -> None:
    cases = [
        TestCase(id="t1", question="Q1?", question_type=QuestionType.FACTUAL),
        TestCase(id="t2", question="Q2?", question_type=QuestionType.ANALYTICAL),
    ]
    suite = TestSuite(name="roundtrip-suite", description="desc", cases=cases)
    restored = TestSuite.model_validate(suite.model_dump())
    assert restored == suite
    assert restored.cases[1].question_type is QuestionType.ANALYTICAL


def test_rag_response_roundtrip() -> None:
    doc = Document(id="d1", content="chunk", metadata={"idx": 0})
    sql = SQLResult(query="SELECT 1", rows=[{"x": 1}])
    resp = RAGResponse(
        answer="A",
        retrieved_docs=[doc],
        sql_result=sql,
        routing_decision=QuestionType.HYBRID,
        latency_ms=100.0,
        cost_usd=0.001,
    )
    restored = RAGResponse.model_validate(resp.model_dump())
    assert restored == resp
    assert restored.sql_result is not None
    assert restored.sql_result.rows[0]["x"] == 1


def test_evaluation_result_roundtrip() -> None:
    now = datetime.now(tz=UTC)
    er = EvaluationResult(
        suite_name="s",
        system_name="sys",
        run_at=now,
        case_results=[_make_case_result()],
        aggregate_scores={"p_at_5": 0.5},
        total_cost_usd=0.01,
        total_duration_seconds=3.0,
        errors=["one error"],
    )
    restored = EvaluationResult.model_validate(er.model_dump())
    assert restored.suite_name == er.suite_name
    assert restored.run_at == er.run_at
    assert restored.errors == ["one error"]
    assert restored.case_results[0].case_id == "c1"
