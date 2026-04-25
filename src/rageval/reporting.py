"""HTML reporting for rageval evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rageval.types import EvaluationResult, MetricResult, QuestionType

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"

_METRIC_DESCRIPTIONS: dict[str, str] = {
    "numeric_tolerance": (
        "Extracts numbers from the answer text and checks the closest one "
        "against the case's expected_numeric within tolerance. Skipped when "
        "the case has no expected numeric."
    ),
    "sql_equivalence": (
        "Set-equality between executed SQL rows and expected_sql_rows; "
        "hybrid cases use containment matching. Skipped when the case has "
        "no expected SQL rows."
    ),
    "refusal": (
        "Did the system refuse appropriately? Scores 1.0 when the system's "
        "refused flag matches the case's should_refuse flag."
    ),
    "prefix_precision@5": (
        "Of the top 5 retrieved chunks, the fraction whose article ID prefix "
        "(before #) matches a relevant article. Skipped when no relevant "
        "doc IDs are set."
    ),
    "prefix_recall@5": (
        "Of the relevant articles, the fraction reached by at least one "
        "retrieved chunk in the top 5 (article-prefix match)."
    ),
    "prefix_ndcg@5": (
        "Rank-aware retrieval quality at k=5 over article ID prefixes."
    ),
    "prefix_reciprocal_rank": (
        "1 divided by the rank of the first retrieved chunk whose article "
        "ID prefix matches a relevant article."
    ),
}

_ROUTE_LABELS: dict[str, str] = {
    QuestionType.FACTUAL.value: "Factual",
    QuestionType.ANALYTICAL.value: "Analytical",
    QuestionType.HYBRID.value: "Hybrid",
    QuestionType.UNANSWERABLE.value: "Unanswerable / Adversarial",
    "missing": "Missing",
}

_CATEGORY_ORDER: list[str] = [
    QuestionType.FACTUAL.value,
    QuestionType.ANALYTICAL.value,
    QuestionType.HYBRID.value,
    QuestionType.UNANSWERABLE.value,
]


def render_html_report(result: EvaluationResult) -> str:
    """Render *result* as a standalone HTML report."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(("html", "xml", "j2")),
    )
    env.filters["json_pretty"] = _json_pretty
    env.filters["money"] = _money
    env.filters["seconds"] = _seconds
    env.filters["short"] = _short
    env.filters["route_label"] = _route_label
    env.filters["route_key"] = _route_key
    env.filters["score_class"] = _score_class
    template = env.get_template(_TEMPLATE_NAME)
    summaries = _metric_summaries(result)
    return template.render(
        result=result,
        metric_summaries=summaries,
        diagnostics=_diagnostics(result),
        route_distribution=_route_distribution(result),
        coverage_data=_coverage_data(summaries),
        category_breakdown=_category_breakdown(result),
        failure_modes=_failure_modes(result),
    )


def _metric_summaries(result: EvaluationResult) -> list[dict[str, Any]]:
    by_metric: dict[str, list[MetricResult]] = {}
    for case_result in result.case_results:
        for metric_result in case_result.metric_results:
            by_metric.setdefault(metric_result.metric_name, []).append(metric_result)

    summaries: list[dict[str, Any]] = []
    for metric_name, metric_results in sorted(by_metric.items()):
        successes = [m for m in metric_results if m.error is None]
        errors = [m for m in metric_results if m.error is not None]
        summaries.append(
            {
                "metric_name": metric_name,
                "description": _METRIC_DESCRIPTIONS.get(metric_name, ""),
                "score": result.aggregate_scores.get(metric_name),
                "successful_count": len(successes),
                "error_count": len(errors),
                "total_count": len(metric_results),
                "skipped_count": max(len(result.case_results) - len(metric_results), 0),
                "case_count": len(result.case_results),
            }
        )
    return summaries


def _diagnostics(result: EvaluationResult) -> dict[str, int]:
    diagnostics: dict[str, int] = {
        "total_cases": len(result.case_results),
        "refused_cases": sum(1 for case in result.case_results if case.response.refused),
        "cases_with_sql": sum(
            1 for case in result.case_results if case.response.sql_result is not None
        ),
        "cases_with_retrieved_docs": sum(
            1 for case in result.case_results if case.response.retrieved_docs
        ),
        "overall_errors": len(result.errors),
        "metric_errors": sum(
            1
            for case in result.case_results
            for metric in case.metric_results
            if metric.error is not None
        ),
    }
    route_counts: dict[str, int] = {}
    for case in result.case_results:
        route = case.response.routing_decision
        route_name = route.value if route is not None else "missing"
        route_counts[route_name] = route_counts.get(route_name, 0) + 1
    for route_name, count in route_counts.items():
        diagnostics[f"route_{route_name}"] = count
    return diagnostics


def _route_distribution(result: EvaluationResult) -> list[dict[str, Any]]:
    counts: dict[str, int] = {key: 0 for key in _ROUTE_LABELS}
    for case in result.case_results:
        route = case.response.routing_decision
        key = route.value if route is not None else "missing"
        counts[key] = counts.get(key, 0) + 1
    distribution: list[dict[str, Any]] = []
    for key, label in _ROUTE_LABELS.items():
        count = counts.get(key, 0)
        if key == "missing" and count == 0:
            continue
        distribution.append({"key": key, "label": label, "count": count})
    return distribution


def _category_breakdown(result: EvaluationResult) -> dict[str, Any]:
    metric_names = sorted(
        {m.metric_name for cr in result.case_results for m in cr.metric_results}
    )

    cases_by_category: dict[str, list[Any]] = {}
    for case in result.case_results:
        key = case.question_type.value if case.question_type is not None else "unknown"
        cases_by_category.setdefault(key, []).append(case)

    rows: list[dict[str, Any]] = []
    ordered_keys = list(_CATEGORY_ORDER) + [
        key for key in cases_by_category if key not in _CATEGORY_ORDER
    ]
    for category in ordered_keys:
        cat_cases = cases_by_category.get(category)
        if not cat_cases:
            continue
        cells: list[dict[str, Any]] = []
        for metric_name in metric_names:
            successful_values: list[float] = []
            error_count = 0
            emitted_count = 0
            for cr in cat_cases:
                for m in cr.metric_results:
                    if m.metric_name != metric_name:
                        continue
                    emitted_count += 1
                    if m.error is None:
                        successful_values.append(m.value)
                    else:
                        error_count += 1
            mean = (
                sum(successful_values) / len(successful_values)
                if successful_values
                else None
            )
            cells.append(
                {
                    "metric_name": metric_name,
                    "mean": mean,
                    "n": len(successful_values),
                    "error_count": error_count,
                    "emitted_count": emitted_count,
                    "skipped_count": len(cat_cases) - emitted_count,
                }
            )
        rows.append(
            {
                "category": category,
                "label": _ROUTE_LABELS.get(category, category.title()),
                "case_count": len(cat_cases),
                "cells": cells,
            }
        )

    return {"metric_names": metric_names, "rows": rows}


def _failure_modes(result: EvaluationResult) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for cr in result.case_results:
        category_key = cr.question_type.value if cr.question_type is not None else "missing"
        category_label = _ROUTE_LABELS.get(category_key, category_key.title())

        if cr.response.routing_decision is None:
            issues.append(
                {
                    "case_id": cr.case_id,
                    "category": category_label,
                    "category_key": category_key,
                    "issue_type": "missing_route",
                    "metric_name": None,
                    "explanation": "No routing decision was produced.",
                }
            )

        sql_result = cr.response.sql_result
        if sql_result is not None and sql_result.error:
            issues.append(
                {
                    "case_id": cr.case_id,
                    "category": category_label,
                    "category_key": category_key,
                    "issue_type": "sql_error",
                    "metric_name": None,
                    "explanation": _short(sql_result.error, 120),
                }
            )

        for m in cr.metric_results:
            if m.error is not None:
                issues.append(
                    {
                        "case_id": cr.case_id,
                        "category": category_label,
                        "category_key": category_key,
                        "issue_type": "metric_error",
                        "metric_name": m.metric_name,
                        "explanation": _short(m.error, 120),
                    }
                )
                continue
            if m.value == 0.0:
                if m.metric_name == "refusal":
                    expected = m.details.get("expected_refused")
                    actual = m.details.get("actual_refused")
                    issues.append(
                        {
                            "case_id": cr.case_id,
                            "category": category_label,
                            "category_key": category_key,
                            "issue_type": "refusal_disagreement",
                            "metric_name": m.metric_name,
                            "explanation": (
                                f"Expected refused={expected}, got refused={actual}."
                            ),
                        }
                    )
                else:
                    issues.append(
                        {
                            "case_id": cr.case_id,
                            "category": category_label,
                            "category_key": category_key,
                            "issue_type": "metric_zero",
                            "metric_name": m.metric_name,
                            "explanation": "Metric scored 0.0.",
                        }
                    )
    return issues


def _coverage_data(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric_name": s["metric_name"],
            "successful": s["successful_count"],
            "errors": s["error_count"],
            "skipped": s["skipped_count"],
        }
        for s in summaries
    ]


def _json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _money(value: float | None) -> str:
    if value is None:
        return "$0.000000"
    return f"${value:.6f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "0.00s"
    return f"{value:.2f}s"


def _short(value: str | None, length: int = 240) -> str:
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= length:
        return normalized
    return f"{normalized[: length - 3]}..."


def _route_key(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, QuestionType):
        return value.value
    return str(value)


def _route_label(value: Any) -> str:
    return _ROUTE_LABELS.get(_route_key(value), "Missing")


def _score_class(value: float | None) -> str:
    if value is None:
        return "score-na"
    if value >= 0.85:
        return "score-good"
    if value >= 0.6:
        return "score-mid"
    return "score-low"
