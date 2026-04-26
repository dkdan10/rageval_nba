import asyncio

import pytest

from rageval.evaluator import Evaluator
from rageval.types import MetricResult, QuestionType, RAGResponse, TestCase, TestSuite


def _suite(num_cases: int = 2) -> TestSuite:
    return TestSuite(
        name="eval-suite",
        cases=[
            TestCase(
                id=f"case-{i}",
                question=f"Question {i}?",
                question_type=QuestionType.FACTUAL,
            )
            for i in range(num_cases)
        ],
    )


class FakeSystem:
    name = "fake-system"

    def __init__(self, cost_usd: float = 0.01) -> None:
        self.cost_usd = cost_usd
        self.questions: list[str] = []

    async def answer(self, question: str) -> RAGResponse:
        self.questions.append(question)
        return RAGResponse(answer=f"answer for {question}", cost_usd=self.cost_usd)


class NamelessSystem:
    async def answer(self, question: str) -> RAGResponse:
        return RAGResponse(answer=question)


class FailingSystem:
    async def answer(self, question: str) -> RAGResponse:  # noqa: ARG002
        raise RuntimeError("boom")


class ConcurrencySystem:
    def __init__(self) -> None:
        self.active = 0
        self.max_seen = 0

    async def answer(self, question: str) -> RAGResponse:
        self.active += 1
        self.max_seen = max(self.max_seen, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return RAGResponse(answer=question)


class SyncMetric:
    metric_name = "sync_score"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        return MetricResult(
            metric_name=self.metric_name,
            case_id=case.id,
            value=1.0 if response.answer else 0.0,
        )


class AsyncEvaluateMetric:
    async def evaluate(self, case: TestCase, response: RAGResponse) -> MetricResult:  # noqa: ARG002
        return MetricResult(metric_name="async_score", case_id=case.id, value=0.5)


class AwaitableCallableMetric:
    metric_name = "awaitable_score"

    async def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:  # noqa: ARG002
        return MetricResult(metric_name=self.metric_name, case_id=case.id, value=0.25)


class ExplodingMetric:
    metric_name = "exploding_score"

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:  # noqa: ARG002
        raise ValueError("metric blew up")


class SkippingMetric:
    metric_name = "skipping_score"

    def __call__(self, case: TestCase, response: RAGResponse) -> None:  # noqa: ARG002
        return None


async def test_evaluate_successful_multiple_cases() -> None:
    system = FakeSystem()
    evaluator = Evaluator(metrics=[SyncMetric()])

    result = await evaluator.evaluate(system, _suite(3))

    assert result.suite_name == "eval-suite"
    assert result.system_name == "fake-system"
    assert len(result.case_results) == 3
    assert system.questions == ["Question 0?", "Question 1?", "Question 2?"]
    assert result.errors == []


async def test_aggregate_scores_mean_successful_metric_results() -> None:
    class VariableMetric:
        metric_name = "variable"

        def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:  # noqa: ARG002
            value = 1.0 if case.id == "case-0" else 0.0
            return MetricResult(metric_name=self.metric_name, case_id=case.id, value=value)

    result = await Evaluator(metrics=[VariableMetric()]).evaluate(FakeSystem(), _suite(2))

    assert result.aggregate_scores == {"variable": 0.5}


async def test_errored_metric_results_are_excluded_from_aggregates() -> None:
    result = await Evaluator(metrics=[SyncMetric(), ExplodingMetric()]).evaluate(
        FakeSystem(), _suite(2)
    )

    assert result.aggregate_scores == {"sync_score": 1.0}
    metric_results = result.case_results[0].metric_results
    failed = next(m for m in metric_results if m.metric_name == "exploding_score")
    assert failed.value == 0.0
    assert failed.error is not None
    assert "metric blew up" in failed.error


async def test_skipped_metric_results_are_excluded_from_aggregates() -> None:
    class SkippedMetric:
        metric_name = "skipped_score"

        def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:  # noqa: ARG002
            return MetricResult(
                metric_name=self.metric_name,
                case_id=case.id,
                value=None,
                details={"skipped": True, "reason": "not applicable"},
            )

    result = await Evaluator(metrics=[SyncMetric(), SkippedMetric()]).evaluate(
        FakeSystem(),
        _suite(1),
    )

    assert result.aggregate_scores == {"sync_score": 1.0}
    skipped = result.case_results[0].metric_results[1]
    assert skipped.metric_name == "skipped_score"
    assert skipped.value is None


async def test_none_metric_result_is_skipped() -> None:
    result = await Evaluator(metrics=[SyncMetric(), SkippingMetric()]).evaluate(
        FakeSystem(),
        _suite(1),
    )

    assert [m.metric_name for m in result.case_results[0].metric_results] == ["sync_score"]
    assert result.aggregate_scores == {"sync_score": 1.0}


async def test_total_cost_and_duration_are_populated() -> None:
    result = await Evaluator(metrics=[]).evaluate(FakeSystem(cost_usd=0.125), _suite(2))

    assert result.total_cost_usd == pytest.approx(0.25)
    assert result.total_duration_seconds >= 0.0


async def test_system_name_falls_back_to_class_name() -> None:
    result = await Evaluator(metrics=[]).evaluate(NamelessSystem(), _suite(1))

    assert result.system_name == "NamelessSystem"


async def test_max_concurrent_is_respected() -> None:
    system = ConcurrencySystem()

    await Evaluator(metrics=[], max_concurrent=2).evaluate(system, _suite(8))

    assert system.max_seen <= 2
    assert system.max_seen > 1


async def test_system_exception_is_captured_without_aborting_suite() -> None:
    result = await Evaluator(metrics=[SyncMetric()]).evaluate(FailingSystem(), _suite(2))

    assert len(result.case_results) == 2
    assert len(result.errors) == 2
    assert result.aggregate_scores == {}
    first = result.case_results[0]
    assert first.response.refused is True
    assert first.metric_results[0].metric_name == "system_error"
    assert first.metric_results[0].error is not None
    assert "system.answer failed" in first.metric_results[0].error


async def test_sync_callable_metric_supported() -> None:
    result = await Evaluator(metrics=[SyncMetric()]).evaluate(FakeSystem(), _suite(1))

    assert result.case_results[0].metric_results[0].metric_name == "sync_score"
    assert result.case_results[0].metric_results[0].value == 1.0


async def test_async_evaluate_metric_supported() -> None:
    result = await Evaluator(metrics=[AsyncEvaluateMetric()]).evaluate(FakeSystem(), _suite(1))

    assert result.case_results[0].metric_results[0].metric_name == "async_score"
    assert result.case_results[0].metric_results[0].value == 0.5


async def test_awaitable_callable_metric_supported() -> None:
    result = await Evaluator(metrics=[AwaitableCallableMetric()]).evaluate(
        FakeSystem(), _suite(1)
    )

    assert result.case_results[0].metric_results[0].metric_name == "awaitable_score"
    assert result.case_results[0].metric_results[0].value == 0.25


async def test_empty_metric_list_behavior() -> None:
    result = await Evaluator(metrics=[]).evaluate(FakeSystem(), _suite(2))

    assert result.aggregate_scores == {}
    assert all(case.metric_results == [] for case in result.case_results)


async def test_on_case_complete_callback_runs_once_per_case() -> None:
    seen: list[str] = []

    result = await Evaluator(metrics=[SyncMetric()]).evaluate(
        FakeSystem(),
        _suite(3),
        on_case_complete=lambda case_result: seen.append(case_result.case_id),
    )

    assert sorted(seen) == ["case-0", "case-1", "case-2"]
    assert sorted(case.case_id for case in result.case_results) == sorted(seen)


async def test_on_case_complete_callback_runs_for_system_failures() -> None:
    seen: list[str] = []

    result = await Evaluator(metrics=[SyncMetric()]).evaluate(
        FailingSystem(),
        _suite(2),
        on_case_complete=lambda case_result: seen.append(case_result.case_id),
    )

    assert sorted(seen) == ["case-0", "case-1"]
    assert len(result.errors) == 2
    assert all(case.response.refused for case in result.case_results)


def test_max_concurrent_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        Evaluator(metrics=[], max_concurrent=0)
