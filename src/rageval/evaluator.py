"""Evaluator orchestrator for running metrics over a test suite."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol, cast

from rageval.types import (
    CaseResult,
    EvaluationResult,
    MetricResult,
    RAGResponse,
    RAGSystem,
    TestCase,
    TestSuite,
)


class _AsyncMetric(Protocol):
    async def evaluate(
        self,
        case: TestCase,
        response: RAGResponse,
    ) -> MetricResult | None: ...


_SyncMetric = Callable[
    [TestCase, RAGResponse],
    MetricResult | None | Awaitable[MetricResult | None],
]
_Metric = _AsyncMetric | _SyncMetric
_CaseCompleteCallback = Callable[[CaseResult], None]


class Evaluator:
    """Run a RAG system against a test suite and aggregate metric results.

    Metric results with ``error`` set are included in each case result, but are
    excluded from aggregate score means so one broken metric call does not pull a
    suite-level score down as if it were a genuine zero.
    """

    def __init__(self, metrics: Sequence[_Metric], max_concurrent: int = 5) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be >= 1")
        self.metrics = list(metrics)
        self.max_concurrent = max_concurrent

    async def evaluate(
        self,
        system: RAGSystem,
        suite: TestSuite,
        *,
        on_case_complete: _CaseCompleteCallback | None = None,
    ) -> EvaluationResult:
        """Evaluate *system* on every case in *suite*."""
        started = time.perf_counter()
        run_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        errors: list[str] = []

        async def run_case(case: TestCase) -> CaseResult:
            async with semaphore:
                case_result = await self._evaluate_case(system, case, errors)
                if on_case_complete is not None:
                    on_case_complete(case_result)
                return case_result

        case_results = await asyncio.gather(*(run_case(case) for case in suite.cases))
        duration = time.perf_counter() - started

        return EvaluationResult(
            suite_name=suite.name,
            system_name=_system_name(system),
            run_at=run_at,
            case_results=list(case_results),
            aggregate_scores=_aggregate_scores(case_results),
            total_cost_usd=sum(
                result.response.cost_usd or 0.0 for result in case_results
            ),
            total_duration_seconds=duration,
            errors=errors,
        )

    async def _evaluate_case(
        self,
        system: RAGSystem,
        case: TestCase,
        errors: list[str],
    ) -> CaseResult:
        try:
            response = await system.answer(case.question)
        except Exception as exc:  # noqa: BLE001 - per-case failures must not abort suite.
            error = f"{case.id}: system.answer failed: {type(exc).__name__}: {exc}"
            errors.append(error)
            response = RAGResponse(answer="", refused=True)
            return CaseResult(
                case_id=case.id,
                question=case.question,
                question_type=case.question_type,
                response=response,
                metric_results=[
                    MetricResult(
                        metric_name="system_error",
                        case_id=case.id,
                        value=0.0,
                        error=error,
                    )
                ],
            )

        metric_results: list[MetricResult] = []
        for metric in self.metrics:
            metric_result = await _run_metric(metric, case, response)
            if metric_result is not None:
                metric_results.append(metric_result)

        return CaseResult(
            case_id=case.id,
            question=case.question,
            question_type=case.question_type,
            response=response,
            metric_results=metric_results,
        )


async def _run_metric(
    metric: _Metric,
    case: TestCase,
    response: RAGResponse,
) -> MetricResult | None:
    metric_name = _metric_name(metric)
    try:
        result: MetricResult | None | Awaitable[MetricResult | None]
        if hasattr(metric, "evaluate"):
            result = cast(_AsyncMetric, metric).evaluate(case, response)
        else:
            result = metric(case, response)

        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001 - metric failures are reported per case.
        return MetricResult(
            metric_name=metric_name,
            case_id=case.id,
            value=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _aggregate_scores(case_results: Sequence[CaseResult]) -> dict[str, float]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for case_result in case_results:
        for metric_result in case_result.metric_results:
            if (
                metric_result.error is None
                and metric_result.value is not None
                and not metric_result.details.get("skipped")
            ):
                values_by_metric[metric_result.metric_name].append(metric_result.value)

    return {
        name: sum(values) / len(values)
        for name, values in values_by_metric.items()
        if values
    }


def _metric_name(metric: object) -> str:
    name = getattr(metric, "metric_name", None)
    if isinstance(name, str) and name:
        return name
    return metric.__class__.__name__


def _system_name(system: object) -> str:
    name = getattr(system, "name", None)
    if isinstance(name, str) and name:
        return name
    return system.__class__.__name__
