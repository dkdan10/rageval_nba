from datetime import UTC, datetime

from rageval.reporting import render_html_report
from rageval.types import (
    CaseResult,
    Document,
    EvaluationResult,
    MetricResult,
    QuestionType,
    RAGResponse,
    SQLResult,
)


def _fake_result() -> EvaluationResult:
    return EvaluationResult(
        suite_name="<script>suite</script>",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="case-001",
                question="What happened?",
                response=RAGResponse(
                    answer="Answer with <b>unsafe</b> text [sql] [article:doc-1]",
                    sql_result=SQLResult(
                        query="SELECT '<tag>' AS value",
                        rows=[{"value": "<tag>"}],
                    ),
                    retrieved_docs=[
                        Document(
                            id="doc-1",
                            content="Document <em>snippet</em>",
                            metadata={
                                "title": "Doc Title",
                                "retrieval_mode": "vector",
                                "score": 0.87654,
                                "distance": 0.12345,
                                "embedding_model": "text-embedding-3-small",
                                "unsafe": "<script>metadata</script>",
                            },
                        )
                    ],
                    routing_decision=QuestionType.HYBRID,
                ),
                metric_results=[
                    MetricResult(
                        metric_name="prefix_recall@5",
                        case_id="case-001",
                        value=0.75,
                        details={"reason": "<unsafe>"},
                    ),
                    MetricResult(
                        metric_name="sql_equivalence",
                        case_id="case-001",
                        value=0.0,
                        error="bad <error>",
                    ),
                ],
            )
        ],
        aggregate_scores={"prefix_recall@5": 0.75},
        total_cost_usd=0.012345,
        total_duration_seconds=1.5,
        errors=["overall <problem>"],
        metadata={"run_mode": "live", "no_cache": True},
    )


def test_render_html_report_contains_summary_and_case_content() -> None:
    html = render_html_report(_fake_result())

    assert "demo-system" in html
    assert "Live evaluation" in html
    assert "Cache bypassed" in html
    assert "case-001" in html
    assert "prefix_recall@5" in html
    assert "0.750" in html
    assert "sql_equivalence" in html
    assert "overall" in html
    assert "SELECT" in html
    assert "doc-1" in html
    assert "Doc Title" in html
    assert "vector" in html
    assert "score 0.877" in html
    assert "distance 0.123" in html
    assert "text-embedding-3-small" in html
    assert "1 successful" in html
    assert "1 emitted" in html
    assert "0 skipped of" in html
    assert "Executive Summary" in html
    assert "Cases" in html
    assert "SQL Evidence" in html
    assert "Retrieved Evidence" in html
    assert "Metric Errors" in html
    assert "Hybrid (SQL + RAG)" in html
    assert "Unanswerable / Adversarial" in html
    assert "CDNs" in html


def test_render_html_report_labels_offline_as_fixture_demo() -> None:
    result = _fake_result().model_copy(update={"metadata": {"run_mode": "offline"}})

    html = render_html_report(result)

    assert "Deterministic fixture demo" in html
    assert "labeled suite routing" in html
    assert "canned/demo SQL branches" in html
    assert "Use --live to exercise the real Router" in html


def test_render_html_report_surfaces_vector_fallback_findings() -> None:
    result = _fake_result()
    fallback_doc = Document(
        id="doc-fallback",
        content="Fallback snippet",
        metadata={
            "title": "Fallback Doc",
            "retrieval_mode": "lexical_fallback",
            "fallback_reason": "embedding_query_failed",
        },
    )
    case = result.case_results[0]
    response = case.response.model_copy(
        update={"retrieved_docs": [fallback_doc]},
    )
    result = result.model_copy(
        update={
            "case_results": [case.model_copy(update={"response": response})],
        }
    )

    html = render_html_report(result)

    assert "Vector Fallbacks" in html
    assert "Vector retrieval fallback" in html
    assert "lexical_fallback" in html
    assert "fallback: embedding_query_failed" in html
    assert "case-001" in html


def test_render_html_report_surfaces_docless_vector_fallback() -> None:
    result = _fake_result()
    case = result.case_results[0]
    response = case.response.model_copy(
        update={
            "retrieved_docs": [],
            "metadata": {
                "retrieval": {
                    "requested_mode": "vector",
                    "retrieval_mode": "lexical_fallback",
                    "fallback_reason": "no_vector_results",
                    "fallback_doc_count": 0,
                }
            },
        }
    )
    result = result.model_copy(
        update={
            "case_results": [case.model_copy(update={"response": response})],
        }
    )

    html = render_html_report(result)

    assert "Vector Fallbacks" in html
    assert "Vector retrieval fallback" in html
    assert "no_vector_results" in html
    assert "case-001" in html


def test_render_html_report_includes_intro_and_metric_descriptions() -> None:
    html = render_html_report(_fake_result())

    assert "Retrieval" in html
    assert "SQL / Structured" in html
    assert "Refusal" in html
    # Metric Coverage chart still labels the section "Coverage".
    assert "Metric Coverage" in html
    assert "Case Browser" in html
    assert "Metric Results" in html
    # The executive summary now leads with the system_name + case-count line
    # instead of restating duration/cost (those live in the status-strip).
    assert "demo-system run evaluated 1 cases" in html
    # Coverage / skipped explanation appears in the methodology notes
    assert "Skipped cases are not failures" in html
    # Metric descriptions appear for known metrics
    assert "Prefix retrieval metrics evaluate article-level IDs" in html
    # Notes / footer details appear
    assert "Methodology Notes" in html


def test_render_html_report_renders_route_badges_and_charts() -> None:
    html = render_html_report(_fake_result())

    # Route badge appears for the hybrid case
    assert "Hybrid (SQL + RAG)" in html
    # Chart payload data is embedded for client-side rendering
    assert "rageval-route-data" in html
    assert "rageval-metric-data" in html
    # Route and coverage chart canvases are present; the dead aggregate-chart
    # canvas was removed.
    assert 'id="route-chart"' in html
    assert 'id="coverage-chart"' in html
    assert 'id="aggregate-chart"' not in html


def test_render_html_report_renders_explicit_metric_skips() -> None:
    result = EvaluationResult(
        suite_name="skip-suite",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="case-001",
                question="Q?",
                question_type=QuestionType.FACTUAL,
                response=RAGResponse(answer="answer"),
                metric_results=[
                    MetricResult(
                        metric_name="sql_equivalence",
                        case_id="case-001",
                        value=None,
                        details={
                            "skipped": True,
                            "reason": "live_expected_sql_rows is not set",
                        },
                    )
                ],
            )
        ],
        aggregate_scores={},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )

    html = render_html_report(result)

    assert "skipped" in html
    assert "live_expected_sql_rows is not set" in html
    assert "Explicit skips" in html


def test_render_html_report_escapes_unsafe_html() -> None:
    html = render_html_report(_fake_result())

    assert "<script>suite</script>" not in html
    assert "&lt;script&gt;suite&lt;/script&gt;" in html
    assert "<b>unsafe</b>" not in html
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in html
    assert "<em>snippet</em>" not in html
    assert "&lt;em&gt;snippet&lt;/em&gt;" in html
    assert "bad &lt;error&gt;" in html
    assert "<script>metadata</script>" not in html
    assert "&lt;script&gt;metadata&lt;/script&gt;" not in html


def test_render_html_report_handles_missing_run_metadata() -> None:
    result = _fake_result().model_copy(update={"metadata": {}})

    html = render_html_report(result)

    assert "Evaluation" in html
    assert "Live evaluation" not in html
    assert "Deterministic fixture demo" not in html


def test_render_html_report_handles_missing_routing_decision() -> None:
    result = EvaluationResult(
        suite_name="missing-routes",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="case-001",
                question="What happened?",
                response=RAGResponse(answer="answer", routing_decision=None),
                metric_results=[],
            )
        ],
        aggregate_scores={},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )

    html = render_html_report(result)

    assert "Missing" in html


def _multi_category_result() -> EvaluationResult:
    return EvaluationResult(
        suite_name="multi-cat",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="f1",
                question="Q?",
                question_type=QuestionType.FACTUAL,
                response=RAGResponse(
                    answer="a", routing_decision=QuestionType.FACTUAL
                ),
                metric_results=[
                    MetricResult(
                        metric_name="numeric_tolerance",
                        case_id="f1",
                        value=1.0,
                    ),
                    MetricResult(metric_name="refusal", case_id="f1", value=1.0),
                ],
            ),
            CaseResult(
                case_id="f2",
                question="Q?",
                question_type=QuestionType.FACTUAL,
                response=RAGResponse(
                    answer="a", routing_decision=QuestionType.FACTUAL
                ),
                metric_results=[
                    MetricResult(
                        metric_name="numeric_tolerance",
                        case_id="f2",
                        value=0.0,
                    ),
                    MetricResult(metric_name="refusal", case_id="f2", value=1.0),
                ],
            ),
            CaseResult(
                case_id="a1",
                question="Q?",
                question_type=QuestionType.ANALYTICAL,
                response=RAGResponse(
                    answer="a", routing_decision=QuestionType.ANALYTICAL
                ),
                metric_results=[
                    MetricResult(
                        metric_name="prefix_recall@5", case_id="a1", value=1.0
                    ),
                    MetricResult(metric_name="refusal", case_id="a1", value=1.0),
                ],
            ),
        ],
        aggregate_scores={"numeric_tolerance": 0.5, "prefix_recall@5": 1.0, "refusal": 1.0},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )


def test_render_html_report_renders_per_category_breakdown() -> None:
    html = render_html_report(_multi_category_result())

    assert "Per-Category Metric Heatmap" in html
    assert "Factual" in html
    assert "Analytical" in html
    # Factual numeric_tolerance mean = (1.0 + 0.0) / 2 = 0.500
    assert "0.500" in html
    # Analytical prefix_recall mean = 1.000 (n=1)
    assert "n = 1" in html
    # Skipped/inapplicable cells render an em dash
    assert "—" in html


def test_render_html_report_uses_adversarial_unanswerable_label() -> None:
    result = EvaluationResult(
        suite_name="adversarial-label",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="adv-001",
                question="Q?",
                question_type=QuestionType.UNANSWERABLE,
                response=RAGResponse(
                    answer="refused",
                    refused=True,
                    routing_decision=QuestionType.UNANSWERABLE,
                ),
                metric_results=[
                    MetricResult(metric_name="refusal", case_id="adv-001", value=1.0)
                ],
            )
        ],
        aggregate_scores={"refusal": 1.0},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )

    html = render_html_report(result)

    assert "Unanswerable / Adversarial" in html


def test_render_html_report_failure_modes_panel_lists_zero_score() -> None:
    html = render_html_report(_multi_category_result())

    assert "Notable Findings" in html
    # Factual case f2 had numeric_tolerance=0.0
    assert "f2" in html
    assert "Zero score" in html


def test_render_html_report_failure_modes_panel_lists_metric_error() -> None:
    html = render_html_report(_fake_result())

    assert "Notable Findings" in html
    assert "Metric error" in html
    assert "case-001" in html


def test_render_html_report_failure_modes_panel_flags_refusal_disagreement() -> None:
    result = EvaluationResult(
        suite_name="refusal",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="adv-001",
                question="Q?",
                question_type=QuestionType.UNANSWERABLE,
                response=RAGResponse(
                    answer="a",
                    refused=False,
                    routing_decision=QuestionType.UNANSWERABLE,
                ),
                metric_results=[
                    MetricResult(
                        metric_name="refusal",
                        case_id="adv-001",
                        value=0.0,
                        details={"expected_refused": True, "actual_refused": False},
                    )
                ],
            )
        ],
        aggregate_scores={"refusal": 0.0},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )

    html = render_html_report(result)

    assert "Refusal disagreement" in html
    assert "Expected refused=True, got refused=False." in html
    # The generic zero-score badge should not be used for the refusal case.
    assert "Zero score" not in html


def test_render_html_report_failure_modes_panel_empty_state() -> None:
    result = EvaluationResult(
        suite_name="clean",
        system_name="demo-system",
        run_at=datetime(2026, 4, 25, tzinfo=UTC),
        case_results=[
            CaseResult(
                case_id="ok-001",
                question="Q?",
                question_type=QuestionType.FACTUAL,
                response=RAGResponse(
                    answer="a", routing_decision=QuestionType.FACTUAL
                ),
                metric_results=[
                    MetricResult(metric_name="refusal", case_id="ok-001", value=1.0)
                ],
            )
        ],
        aggregate_scores={"refusal": 1.0},
        total_cost_usd=0.0,
        total_duration_seconds=0.1,
    )

    html = render_html_report(result)

    assert "No highlighted findings." in html
