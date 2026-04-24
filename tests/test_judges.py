"""Tests for Milestone 4: FaithfulnessJudge."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from rageval.metrics.judge import _MODEL, FaithfulnessJudge
from rageval.types import Document, QuestionType, RAGResponse, SQLResult, TestCase

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "faithfulness_calibration.yaml"

_FAITHFUL_JSON = (
    '{"reasoning": "all claims are supported", "faithful": true, "unsupported_claims": []}'
)
_UNFAITHFUL_JSON = (
    '{"reasoning": "player name is wrong", "faithful": false,'
    ' "unsupported_claims": ["Stephen Curry led in scoring"]}'
)


def _make_case(case_id: str = "test-001") -> TestCase:
    return TestCase(
        id=case_id,
        question="Who led the NBA in points per game?",
        question_type=QuestionType.FACTUAL,
    )


def _make_response(
    answer: str = "Jayson Tatum led with 30.1 PPG.",
    sql_result: SQLResult | None = None,
    retrieved_docs: list[Document] | None = None,
) -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sql_result=sql_result,
        retrieved_docs=retrieved_docs or [],
    )


def _make_judge(llm_content: str) -> FaithfulnessJudge:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_content})
    return FaithfulnessJudge(llm=llm)


# ---------------------------------------------------------------------------
# Core scoring tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_faithful_output_returns_1() -> None:
    judge = _make_judge(_FAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.metric_name == "faithfulness"
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_unfaithful_output_returns_0() -> None:
    judge = _make_judge(_UNFAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is None


@pytest.mark.asyncio
async def test_unsupported_claims_in_details() -> None:
    judge = _make_judge(_UNFAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert "unsupported_claims" in result.details
    claims: Any = result.details["unsupported_claims"]
    assert isinstance(claims, list)
    assert len(claims) == 1
    assert "Stephen Curry" in claims[0]


# ---------------------------------------------------------------------------
# Error / malformed output tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_returns_0_with_error() -> None:
    judge = _make_judge("this is not json at all")
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "Invalid JSON" in result.error


@pytest.mark.asyncio
async def test_missing_faithful_field_returns_0_with_error() -> None:
    judge = _make_judge('{"reasoning": "looks good", "unsupported_claims": []}')
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "faithful" in result.error


@pytest.mark.asyncio
async def test_malformed_unsupported_claims_handled_safely() -> None:
    content = '{"reasoning": "ok", "faithful": false, "unsupported_claims": "not a list"}'
    judge = _make_judge(content)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is None
    claims: Any = result.details["unsupported_claims"]
    assert isinstance(claims, list)
    assert len(claims) == 1


# ---------------------------------------------------------------------------
# Source inclusion in user prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_sources_included_in_prompt() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _FAITHFUL_JSON})
    judge = FaithfulnessJudge(llm=llm)
    sql = SQLResult(query="SELECT player, ppg FROM stats", rows=[{"player": "Tatum", "ppg": 30.1}])
    await judge.evaluate(_make_case(), _make_response(sql_result=sql))
    user_prompt: str = llm.complete.call_args.kwargs["user"]
    assert "SELECT player, ppg FROM stats" in user_prompt
    assert "Tatum" in user_prompt


@pytest.mark.asyncio
async def test_retrieved_docs_included_in_prompt() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _FAITHFUL_JSON})
    judge = FaithfulnessJudge(llm=llm)
    doc = Document(id="doc-nba-1", content="LeBron James scored 28 points in Game 7.")
    await judge.evaluate(_make_case(), _make_response(retrieved_docs=[doc]))
    user_prompt: str = llm.complete.call_args.kwargs["user"]
    assert "doc-nba-1" in user_prompt
    assert "LeBron James scored 28 points in Game 7." in user_prompt


@pytest.mark.asyncio
async def test_both_sources_included_in_prompt() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _FAITHFUL_JSON})
    judge = FaithfulnessJudge(llm=llm)
    sql = SQLResult(query="SELECT team FROM standings", rows=[{"team": "Celtics"}])
    doc = Document(id="doc-nba-2", content="Golden State Warriors won in 2022.")
    await judge.evaluate(_make_case(), _make_response(sql_result=sql, retrieved_docs=[doc]))
    user_prompt: str = llm.complete.call_args.kwargs["user"]
    assert "SELECT team FROM standings" in user_prompt
    assert "Golden State Warriors" in user_prompt


# ---------------------------------------------------------------------------
# LLMClient.complete call arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_called_with_correct_model_and_temperature() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _FAITHFUL_JSON})
    judge = FaithfulnessJudge(llm=llm)
    await judge.evaluate(_make_case(), _make_response())
    llm.complete.assert_called_once()
    kwargs: dict[str, Any] = llm.complete.call_args.kwargs
    assert kwargs["model"] == _MODEL
    assert kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Calibration fixture tests
# ---------------------------------------------------------------------------


def test_calibration_yaml_has_10_cases() -> None:
    data: Any = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases: list[Any] = data["cases"]
    assert len(cases) == 10


def test_calibration_yaml_has_both_labels() -> None:
    data: Any = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases: list[Any] = data["cases"]
    labels: list[Any] = [c["human_label"] for c in cases]
    assert True in labels
    assert False in labels
