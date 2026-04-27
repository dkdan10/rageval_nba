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

_ROUTE_DISPLAY_LABELS: dict[str, str] = {
    QuestionType.FACTUAL.value: "SQL Only",
    QuestionType.ANALYTICAL.value: "RAG Only",
    QuestionType.HYBRID.value: "Hybrid (SQL + RAG)",
    QuestionType.UNANSWERABLE.value: "Refusal",
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
    env.filters["route_display_label"] = _route_display_label
    env.filters["score_class"] = _score_class
    env.filters["score_word"] = _score_word
    env.filters["run_mode_label"] = _run_mode_label
    template = env.get_template(_TEMPLATE_NAME)
    summaries = _metric_summaries(result)
    diagnostics = _diagnostics(result)
    failures = _failure_modes(result)
    case_index_rows = _case_index_rows(result)
    case_status_by_id = {row["case_id"]: row["status"] for row in case_index_rows}
    return template.render(
        result=result,
        metric_summaries=summaries,
        metric_groups=_metric_groups(summaries),
        metric_scorecards=_metric_scorecards(summaries, diagnostics),
        diagnostics=diagnostics,
        health_summary=_health_summary(result, diagnostics),
        run_summary=_run_summary(result, diagnostics, summaries),
        executive_summary_lines=_executive_summary_lines(result, diagnostics, summaries),
        route_distribution=_route_distribution(result),
        coverage_data=_coverage_data(summaries),
        category_breakdown=_category_breakdown(result),
        failure_modes=failures,
        notable_findings=_notable_findings(failures),
        finding_cards=_finding_cards(failures, diagnostics),
        case_index_rows=case_index_rows,
        case_status_by_id=case_status_by_id,
        run_metadata=_run_metadata(result),
    )


def _metric_summaries(result: EvaluationResult) -> list[dict[str, Any]]:
    by_metric: dict[str, list[MetricResult]] = {}
    for case_result in result.case_results:
        for metric_result in case_result.metric_results:
            by_metric.setdefault(metric_result.metric_name, []).append(metric_result)

    summaries: list[dict[str, Any]] = []
    for metric_name, metric_results in sorted(by_metric.items()):
        explicit_skips = [m for m in metric_results if m.details.get("skipped")]
        successes = [
            m
            for m in metric_results
            if m.error is None and m.value is not None and not m.details.get("skipped")
        ]
        errors = [m for m in metric_results if m.error is not None]
        skipped_count = len(explicit_skips) + max(
            len(result.case_results) - len(metric_results),
            0,
        )
        summaries.append(
            {
                "metric_name": metric_name,
                "description": _METRIC_DESCRIPTIONS.get(metric_name, ""),
                "score": result.aggregate_scores.get(metric_name),
                "successful_count": len(successes),
                "error_count": len(errors),
                "total_count": len(metric_results),
                "skipped_count": skipped_count,
                "explicit_skipped_count": len(explicit_skips),
                "case_count": len(result.case_results),
            }
        )
    return summaries


def _metric_family(metric_name: str) -> str:
    if metric_name.startswith("prefix_") or metric_name in {
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "reciprocal_rank",
    }:
        return "Retrieval"
    if metric_name in {"sql_equivalence", "numeric_tolerance", "exact_match"}:
        return "SQL / Structured"
    if metric_name == "refusal":
        return "Refusal"
    if metric_name in {"faithfulness", "relevance", "correctness", "routing"}:
        return "Judge / Quality"
    return "Other"


def _metric_groups(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["Retrieval", "SQL / Structured", "Refusal", "Judge / Quality", "Other"]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_family.setdefault(_metric_family(str(summary["metric_name"])), []).append(summary)
    return [
        {"family": family, "metrics": by_family[family]}
        for family in order
        if by_family.get(family)
    ]


def _summary_by_name(
    summaries: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    return next((summary for summary in summaries if summary["metric_name"] == name), None)


def _metric_display(summary: dict[str, Any] | None, fallback_name: str) -> dict[str, Any]:
    if summary is None:
        return {
            "metric_name": fallback_name,
            "score": None,
            "label": "n/a",
            "subtext": "Not evaluated",
            "quality": "n/a",
        }
    score = summary["score"]
    return {
        "metric_name": summary["metric_name"],
        "score": score,
        "label": f"{score:.3f}" if score is not None else "n/a",
        "subtext": _metric_subtext(summary),
        "quality": _score_word(score),
    }


def _metric_subtext(summary: dict[str, Any]) -> str:
    if summary["metric_name"] == "sql_equivalence":
        base = f"{summary['successful_count']} / {summary['successful_count']}"
        if summary["explicit_skipped_count"]:
            return f"{base}; {summary['explicit_skipped_count']} skipped"
        return base
    if summary["metric_name"] == "refusal" and summary["score"] == 1.0:
        return "Perfect"
    return (
        f"{summary['successful_count']} successful · "
        f"{summary['total_count']} emitted · "
        f"{summary['skipped_count']} skipped of {summary['case_count']}"
    )


def _metric_scorecards(
    summaries: list[dict[str, Any]],
    diagnostics: dict[str, int],
) -> list[dict[str, Any]]:
    cards = [
        {
            "family": "Retrieval",
            "icon": "search",
            "accent": "blue",
            "metrics": [
                _metric_display(summary, "prefix_recall@5")
                for summary in [_summary_by_name(summaries, "prefix_recall@5")]
                if summary is not None
            ]
            + [
                _metric_display(summary, "prefix_ndcg@5")
                for summary in [_summary_by_name(summaries, "prefix_ndcg@5")]
                if summary is not None
            ],
            "hint": "Higher is better",
        },
        {
            "family": "SQL / Structured",
            "icon": "database",
            "accent": "purple",
            "metrics": [
                _metric_display(summary, "sql_equivalence")
                for summary in [_summary_by_name(summaries, "sql_equivalence")]
                if summary is not None
            ]
            + [
                _metric_display(
                    summary,
                    "numeric_tolerance",
                )
                for summary in [_summary_by_name(summaries, "numeric_tolerance")]
                if summary is not None
            ],
            "hint": "Higher is better",
        },
        {
            "family": "Refusal",
            "icon": "shield",
            "accent": "neutral",
            "metrics": [
                _metric_display(summary, "refusal")
                for summary in [_summary_by_name(summaries, "refusal")]
                if summary is not None
            ],
            "hint": "Higher is better",
        },
    ]
    for card in cards:
        card["accent"] = "neutral"
    return [card for card in cards if card["metrics"]]


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


def _health_summary(result: EvaluationResult, diagnostics: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"label": "Cases", "value": str(diagnostics["total_cases"]), "status": "neutral"},
        {
            "label": "Duration",
            "value": _seconds(result.total_duration_seconds),
            "status": "neutral",
        },
        {"label": "Cost", "value": _money(result.total_cost_usd), "status": "neutral"},
        {
            "label": "Overall Errors",
            "value": str(diagnostics["overall_errors"]),
            "status": "good" if diagnostics["overall_errors"] == 0 else "bad",
        },
        {
            "label": "Metric Errors",
            "value": str(diagnostics["metric_errors"]),
            "status": "good" if diagnostics["metric_errors"] == 0 else "bad",
        },
        {"label": "Refused", "value": str(diagnostics["refused_cases"]), "status": "neutral"},
        {"label": "With SQL", "value": str(diagnostics["cases_with_sql"]), "status": "neutral"},
        {
            "label": "With Retrieved Docs",
            "value": str(diagnostics["cases_with_retrieved_docs"]),
            "status": "neutral",
        },
    ]


def _run_summary(
    result: EvaluationResult,
    diagnostics: dict[str, int],
    summaries: list[dict[str, Any]],
) -> str:
    metric_bits: list[str] = []
    for name in ("sql_equivalence", "prefix_recall@5", "refusal"):
        summary = next((s for s in summaries if s["metric_name"] == name), None)
        if summary and summary["score"] is not None:
            if name == "sql_equivalence":
                metric_bits.append(
                    f"SQL equivalence {summary['score']:.3f} on "
                    f"{summary['successful_count']} scored cases"
                )
            elif name == "prefix_recall@5":
                metric_bits.append(f"retrieval recall@5 {summary['score']:.3f}")
            elif name == "refusal":
                metric_bits.append(f"refusal {summary['score']:.3f}")

    metrics_text = "; ".join(metric_bits) if metric_bits else "no aggregate metrics"
    return (
        f"{diagnostics['total_cases']} cases completed in "
        f"{_seconds(result.total_duration_seconds)} with "
        f"{diagnostics['overall_errors']} overall errors and "
        f"{diagnostics['metric_errors']} metric errors; {metrics_text}."
    )


def _executive_summary_lines(
    result: EvaluationResult,
    diagnostics: dict[str, int],
    summaries: list[dict[str, Any]],
) -> list[str]:
    sql = _summary_by_name(summaries, "sql_equivalence")
    recall = _summary_by_name(summaries, "prefix_recall@5")
    ndcg = _summary_by_name(summaries, "prefix_ndcg@5")
    refusal = _summary_by_name(summaries, "refusal")
    numeric = _summary_by_name(summaries, "numeric_tolerance")

    lines = [
        (
            f"The {result.system_name} run evaluated {diagnostics['total_cases']} cases "
            "across factual, analytical, hybrid, and unanswerable/adversarial questions."
        )
    ]
    if sql and sql["score"] is not None:
        skipped = sql["explicit_skipped_count"]
        skip_text = f", {skipped} skipped" if skipped else ""
        lines.append(
            f"SQL equivalence scored {sql['score']:.3f} on evaluated cases "
            f"({sql['successful_count']}/{sql['successful_count']}{skip_text})."
        )
    if recall and recall["score"] is not None:
        retrieval = f"Retrieval reached prefix_recall@5 = {recall['score']:.3f}"
        if ndcg and ndcg["score"] is not None:
            retrieval += f" and ndcg@5 = {ndcg['score']:.3f}"
        lines.append(f"{retrieval}.")
    if refusal and refusal["score"] is not None:
        lines.append(f"Refusal behavior scored {refusal['score']:.3f}.")
    if numeric and numeric["score"] is not None and numeric["score"] < 0.8:
        lines.append(
            f"Numeric tolerance is the main structured area to inspect "
            f"({numeric['score']:.3f})."
        )
    if diagnostics["overall_errors"] == 0 and diagnostics["metric_errors"] == 0:
        lines.append("No overall or metric errors were recorded.")
    return lines


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
        distribution.append(
            {
                "key": key,
                "label": label,
                "display_label": _ROUTE_DISPLAY_LABELS.get(key, label),
                "count": count,
                "percent": count / len(result.case_results) if result.case_results else 0.0,
            }
        )
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
                    if m.details.get("skipped"):
                        continue
                    if m.error is None and m.value is not None:
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
            if m.details.get("skipped"):
                issues.append(
                    {
                        "case_id": cr.case_id,
                        "category": category_label,
                        "category_key": category_key,
                        "issue_type": "metric_skipped",
                        "metric_name": m.metric_name,
                        "explanation": _short(str(m.details.get("reason")), 120),
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


def _finding_group(issue_type: str) -> str:
    if issue_type in {"metric_error", "sql_error", "missing_route"}:
        return "Errors"
    if issue_type == "refusal_disagreement":
        return "Disagreements"
    if issue_type == "metric_skipped":
        return "Explicit Skips"
    if issue_type == "metric_zero":
        return "Zero Scores"
    return "Other"


def _notable_findings(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["Errors", "Disagreements", "Explicit Skips", "Zero Scores", "Other"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        grouped.setdefault(_finding_group(str(issue["issue_type"])), []).append(issue)
    return [
        {"group": group, "issues": grouped[group], "count": len(grouped[group])}
        for group in order
        if grouped.get(group)
    ]


def _finding_cards(
    issues: list[dict[str, Any]],
    diagnostics: dict[str, int],
) -> list[dict[str, Any]]:
    explicit = [issue for issue in issues if issue["issue_type"] == "metric_skipped"]
    zero_retrieval = [
        issue for issue in issues
        if issue["issue_type"] == "metric_zero"
        and str(issue.get("metric_name", "")).startswith("prefix_")
    ]
    metric_errors = [
        issue for issue in issues
        if issue["issue_type"] in {"metric_error", "sql_error", "missing_route"}
    ]
    cards: list[dict[str, Any]] = []
    if explicit:
        cards.append(
            {
                "title": "Explicit skips",
                "description": _summarize_cases(explicit, "case skipped a metric."),
                "count": len(explicit),
                "status": "warn",
                "icon": "!",
            }
        )
    if zero_retrieval:
        cards.append(
            {
                "title": "Zero-score retrieval cases",
                "description": _summarize_cases(zero_retrieval, "case had a retrieval miss."),
                "count": len({issue["case_id"] for issue in zero_retrieval}),
                "status": "err",
                "icon": "×",
            }
        )
    refusal_disagreements = [
        issue for issue in issues if issue["issue_type"] == "refusal_disagreement"
    ]
    if refusal_disagreements:
        cards.append(
            {
                "title": "Refusal disagreement",
                "description": _summarize_cases(
                    refusal_disagreements,
                    "case had refused/should_refuse disagreement.",
                ),
                "count": len(refusal_disagreements),
                "status": "err",
                "icon": "×",
            }
        )
    zero_other = [
        issue
        for issue in issues
        if issue["issue_type"] == "metric_zero"
        and not str(issue.get("metric_name", "")).startswith("prefix_")
    ]
    if zero_other:
        cards.append(
            {
                "title": "Zero score cases",
                "description": _summarize_cases(zero_other, "case had a zero metric score."),
                "count": len({issue["case_id"] for issue in zero_other}),
                "status": "warn",
                "icon": "!",
            }
        )
    cards.append(
        {
            "title": "No metric errors" if diagnostics["metric_errors"] == 0 else "Metric errors",
            "description": (
                "All metrics executed successfully."
                if diagnostics["metric_errors"] == 0
                else _summarize_cases(metric_errors, "case had a metric or SQL error.")
            ),
            "count": diagnostics["metric_errors"],
            "status": "ok" if diagnostics["metric_errors"] == 0 else "err",
            "icon": "✓",
        }
    )
    return cards


def _summarize_cases(issues: list[dict[str, Any]], fallback: str) -> str:
    case_ids = sorted({str(issue["case_id"]) for issue in issues})
    if not case_ids:
        return fallback
    shown = ", ".join(case_ids[:3])
    suffix = "..." if len(case_ids) > 3 else ""
    return f"{len(case_ids)} {fallback} See cases: {shown}{suffix}"


def _case_index_rows(result: EvaluationResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cr in result.case_results:
        route_key = _route_key(cr.response.routing_decision)
        category_key = cr.question_type.value if cr.question_type is not None else "missing"
        metric_errors = [m for m in cr.metric_results if m.error is not None]
        metric_skips = [m for m in cr.metric_results if m.details.get("skipped")]
        zero_scores = [
            m for m in cr.metric_results
            if m.error is None and not m.details.get("skipped") and m.value == 0.0
        ]
        key_scores = _case_key_scores(cr.metric_results)
        flags: list[dict[str, str]] = []
        if cr.response.refused and category_key == QuestionType.UNANSWERABLE.value:
            flags.append({"label": "Correct refusal", "status": "ok"})
        elif cr.response.refused:
            flags.append({"label": "Refused", "status": "warn"})
        if metric_errors:
            flags.append({"label": "Metric error", "status": "err"})
        if any(m.metric_name == "sql_equivalence" for m in metric_skips):
            flags.append({"label": "Skipped SQL", "status": "warn"})
        elif metric_skips:
            flags.append({"label": f"{len(metric_skips)} skip", "status": "warn"})
        if any(str(m.metric_name).startswith("prefix_") for m in zero_scores):
            flags.append({"label": "Retrieval miss", "status": "err"})
        elif zero_scores:
            flags.append({"label": f"{len(zero_scores)} zero", "status": "warn"})
        if not flags:
            flags.append({"label": "Passing", "status": "ok"})

        statuses = {flag["status"] for flag in flags}
        if "err" in statuses:
            row_status = "err"
        elif "warn" in statuses:
            row_status = "warn"
        else:
            row_status = "ok"

        question = cr.question
        rows.append(
            {
                "case_id": cr.case_id,
                "question": question,
                "category_key": category_key,
                "category_label": _ROUTE_LABELS.get(category_key, category_key.title()),
                "route_key": route_key,
                "route_label": _ROUTE_DISPLAY_LABELS.get(route_key, "Missing"),
                "refused": cr.response.refused,
                "key_scores": key_scores,
                "error_count": len(metric_errors),
                "skip_count": len(metric_skips),
                "flags": flags,
                "status": row_status,
            }
        )
    return rows


def _case_key_scores(metric_results: list[MetricResult]) -> list[dict[str, Any]]:
    by_name = {metric.metric_name: metric for metric in metric_results}
    ordered = [
        "prefix_recall@5",
        "sql_equivalence",
        "numeric_tolerance",
        "refusal",
    ]
    scores: list[dict[str, Any]] = []
    for name in ordered:
        metric = by_name.get(name)
        if metric is None:
            continue
        scores.append(
            {
                "metric_name": metric.metric_name,
                "value": metric.value,
                "skipped": bool(metric.details.get("skipped")),
                "error": metric.error,
                "missing": False,
            }
        )
    return scores


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


def _run_metadata(result: EvaluationResult) -> dict[str, Any]:
    run_mode = result.metadata.get("run_mode")
    no_cache = result.metadata.get("no_cache")
    metrics_selected = result.metadata.get("metrics_selected")
    return {
        "run_mode": run_mode if isinstance(run_mode, str) else None,
        "run_mode_label": _run_mode_label(run_mode),
        "no_cache": no_cache if isinstance(no_cache, bool) else None,
        "metrics_selected": metrics_selected if isinstance(metrics_selected, list) else None,
    }


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


def _route_display_label(value: Any) -> str:
    return _ROUTE_DISPLAY_LABELS.get(_route_key(value), "Missing")


def _run_mode_label(value: Any) -> str:
    if value == "live":
        return "Live evaluation"
    if value == "offline":
        return "Offline evaluation"
    return "Evaluation"


def _score_class(value: float | None) -> str:
    if value is None:
        return "score-na"
    if value >= 0.85:
        return "score-good"
    if value >= 0.6:
        return "score-mid"
    return "score-low"


def _score_word(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 0.95:
        return "perfect"
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "fair"
    return "poor"
