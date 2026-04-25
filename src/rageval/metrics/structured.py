"""Deterministic structured metrics for SQL-backed evaluation cases."""

from __future__ import annotations

import re
from typing import Any

from rageval.types import MetricResult, QuestionType, RAGResponse, TestCase

_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _numbers_from_text(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        is_percent = token.endswith("%")
        normalized = token.rstrip("%").replace(",", "")
        try:
            value = float(normalized)
        except ValueError:
            continue
        values.append(value / 100.0 if is_percent else value)
    return values


def _row_exact_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return expected == actual


def _row_contains(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())


def _rows_match(
    expected_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
    *,
    contains: bool,
) -> bool:
    if not contains and len(expected_rows) != len(actual_rows):
        return False

    unmatched = list(actual_rows)
    matcher = _row_contains if contains else _row_exact_match
    for expected in expected_rows:
        match_index = next(
            (i for i, actual in enumerate(unmatched) if matcher(expected, actual)),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return True


class ExactMatchMetric:
    metric_name = "exact_match"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        if case.expected_answer is None:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error="No expected answer available",
            )

        expected = _normalize_text(case.expected_answer)
        actual = _normalize_text(response.answer)
        return MetricResult(
            metric_name=self.metric_name,
            case_id=case.id,
            value=1.0 if actual == expected else 0.0,
            details={"expected_normalized": expected, "actual_normalized": actual},
        )


class NumericToleranceMetric:
    metric_name = "numeric_tolerance"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        if case.expected_numeric is None:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error="No expected numeric value available",
            )

        candidates = _numbers_from_text(response.answer)
        if not candidates:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error="No numeric value found in answer",
            )

        expected = case.expected_numeric
        tolerance = case.numeric_tolerance
        best = min(candidates, key=lambda value: abs(value - expected))
        delta = abs(best - expected)
        return MetricResult(
            metric_name=self.metric_name,
            case_id=case.id,
            value=1.0 if delta <= tolerance else 0.0,
            details={
                "expected_numeric": expected,
                "numeric_tolerance": tolerance,
                "matched_numeric": best,
                "delta": delta,
                "candidates": candidates,
            },
        )


class SQLEquivalenceMetric:
    metric_name = "sql_equivalence"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        if case.expected_sql_rows is None:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error="No expected SQL rows available",
            )
        if response.sql_result is None:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error="No SQL result available",
            )
        if response.sql_result.error is not None:
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=0.0,
                error=f"SQL result error: {response.sql_result.error}",
            )

        contains = case.question_type is QuestionType.HYBRID
        matched = _rows_match(
            case.expected_sql_rows,
            response.sql_result.rows,
            contains=contains,
        )
        return MetricResult(
            metric_name=self.metric_name,
            case_id=case.id,
            value=1.0 if matched else 0.0,
            details={
                "expected_rows": case.expected_sql_rows,
                "actual_rows": response.sql_result.rows,
                "contains_match": contains,
            },
        )


class RefusalMetric:
    metric_name = "refusal"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        expected = case.should_refuse
        actual = response.refused
        return MetricResult(
            metric_name=self.metric_name,
            case_id=case.id,
            value=1.0 if actual == expected else 0.0,
            details={"expected_refused": expected, "actual_refused": actual},
        )
