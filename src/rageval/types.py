from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Self

import yaml
from pydantic import BaseModel, Field, ValidationError


class QuestionType(str, Enum):  # noqa: UP042 — spec calls for `str, Enum` (see PROJECT_PLAN.md)
    """Routing category for a test case."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    UNANSWERABLE = "unanswerable"


class Document(BaseModel):
    """A retrievable chunk of text."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLResult(BaseModel):
    """Output of a SQL path."""

    query: str
    rows: list[dict[str, Any]]
    error: str | None = None


class TestCase(BaseModel):
    """A single evaluation case."""

    id: str
    question: str
    question_type: QuestionType

    expected_sql_rows: list[dict[str, Any]] | None = None
    expected_numeric: float | None = None
    numeric_tolerance: float = 0.01

    relevant_doc_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    should_refuse: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestSuite(BaseModel):
    """A collection of test cases + metadata."""

    name: str
    description: str = ""
    cases: list[TestCase]

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Load and validate a TestSuite from a YAML file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML is malformed or fails schema validation.
        """
        file_path = Path(path)
        try:
            raw = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Test suite file not found: {path}") from None

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at the top level in {path}, "
                f"got {type(data).__name__}"
            )

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid test suite structure in {path}:\n{exc}") from exc


class RAGResponse(BaseModel):
    """What a system-under-test returns."""

    answer: str
    retrieved_docs: list[Document] = Field(default_factory=list)
    sql_result: SQLResult | None = None
    routing_decision: QuestionType | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    refused: bool = False


class RAGSystem(Protocol):
    """Protocol implemented by systems-under-test."""

    async def answer(self, question: str) -> RAGResponse: ...


class MetricResult(BaseModel):
    """Score from running one metric on one case."""

    metric_name: str
    case_id: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CaseResult(BaseModel):
    """Aggregated results for one test case."""

    case_id: str
    question: str
    question_type: QuestionType | None = None
    response: RAGResponse
    metric_results: list[MetricResult]


class EvaluationResult(BaseModel):
    """Complete results of an evaluation run."""

    suite_name: str
    system_name: str
    run_at: datetime
    case_results: list[CaseResult]
    aggregate_scores: dict[str, float]
    total_cost_usd: float
    total_duration_seconds: float
    errors: list[str] = Field(default_factory=list)
