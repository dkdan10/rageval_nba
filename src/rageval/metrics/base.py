from typing import Protocol

from rageval.types import MetricResult, RAGResponse, TestCase


class Metric(Protocol):
    """Protocol for evaluation metrics."""

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult: ...
