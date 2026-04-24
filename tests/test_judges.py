"""Tests for Milestone 4 & 5: Judge suite."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from rageval.metrics.judge import (
    _CORRECTNESS_TOOL,
    _FAITHFULNESS_TOOL,
    _MODEL,
    _RELEVANCE_TOOL,
    CorrectnessJudge,
    FaithfulnessJudge,
    RelevanceJudge,
    RoutingJudge,
)
from rageval.types import Document, QuestionType, RAGResponse, SQLResult, TestCase

FAITH_FIXTURE = Path(__file__).parent / "fixtures" / "faithfulness_calibration.yaml"
RELEVANCE_FIXTURE = Path(__file__).parent / "fixtures" / "relevance_calibration.yaml"
CORRECTNESS_FIXTURE = Path(__file__).parent / "fixtures" / "correctness_calibration.yaml"
ROUTING_FIXTURE = Path(__file__).parent / "fixtures" / "routing_calibration.yaml"

_FAITHFUL_JSON = (
    '{"reasoning": "all claims are supported", "faithful": true, "unsupported_claims": []}'
)
_UNFAITHFUL_JSON = (
    '{"reasoning": "player name is wrong", "faithful": false,'
    ' "unsupported_claims": ["Stephen Curry led in scoring"]}'
)
_RELEVANT_JSON = (
    '{"reasoning": "answer addresses the question", "relevant": true, "irrelevant_parts": []}'
)
_IRRELEVANT_JSON = (
    '{"reasoning": "answer ignores the question", "relevant": false,'
    ' "irrelevant_parts": ["discussion of NBA history"]}'
)
_SCORE_4_JSON = '{"reasoning": "fully correct", "score": 4, "errors": []}'
_SCORE_2_JSON = '{"reasoning": "partially correct", "score": 2, "errors": ["missing detail"]}'
_SCORE_0_JSON = '{"reasoning": "completely wrong", "score": 0, "errors": ["wrong player named"]}'


def _make_case(
    case_id: str = "test-001",
    question_type: QuestionType = QuestionType.FACTUAL,
    expected_answer: str | None = None,
) -> TestCase:
    return TestCase(
        id=case_id,
        question="Who led the NBA in points per game?",
        question_type=question_type,
        expected_answer=expected_answer,
    )


def _make_response(
    answer: str = "Jayson Tatum led with 30.1 PPG.",
    sql_result: SQLResult | None = None,
    retrieved_docs: list[Document] | None = None,
    routing_decision: QuestionType | None = None,
) -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sql_result=sql_result,
        retrieved_docs=retrieved_docs or [],
        routing_decision=routing_decision,
    )


def _make_faith_judge(llm_content: str) -> FaithfulnessJudge:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_content})
    return FaithfulnessJudge(llm=llm)


def _make_relevance_judge(llm_content: str) -> RelevanceJudge:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_content})
    return RelevanceJudge(llm=llm)


def _make_correctness_judge(*llm_contents: str) -> CorrectnessJudge:
    llm = MagicMock()
    if len(llm_contents) == 1:
        llm.complete = AsyncMock(return_value={"content": llm_contents[0]})
    else:
        llm.complete = AsyncMock(
            side_effect=[{"content": c} for c in llm_contents]
        )
    return CorrectnessJudge(llm=llm)


# ===========================================================================
# FaithfulnessJudge
# ===========================================================================


@pytest.mark.asyncio
async def test_faithful_output_returns_1() -> None:
    judge = _make_faith_judge(_FAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.metric_name == "faithfulness"
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_unfaithful_output_returns_0() -> None:
    judge = _make_faith_judge(_UNFAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is None


@pytest.mark.asyncio
async def test_unsupported_claims_in_details() -> None:
    judge = _make_faith_judge(_UNFAITHFUL_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert "unsupported_claims" in result.details
    claims: Any = result.details["unsupported_claims"]
    assert isinstance(claims, list)
    assert len(claims) == 1
    assert "Stephen Curry" in claims[0]


@pytest.mark.asyncio
async def test_faith_invalid_json_returns_error() -> None:
    judge = _make_faith_judge("this is not json at all")
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "Invalid JSON" in result.error


@pytest.mark.asyncio
async def test_missing_faithful_field_returns_error() -> None:
    judge = _make_faith_judge('{"reasoning": "looks good", "unsupported_claims": []}')
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "faithful" in result.error


@pytest.mark.asyncio
async def test_non_bool_faithful_returns_error() -> None:
    judge = _make_faith_judge('{"reasoning": "ok", "faithful": "false", "unsupported_claims": []}')
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None


@pytest.mark.asyncio
async def test_malformed_unsupported_claims_handled_safely() -> None:
    content = '{"reasoning": "ok", "faithful": false, "unsupported_claims": "not a list"}'
    judge = _make_faith_judge(content)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is None
    claims: Any = result.details["unsupported_claims"]
    assert isinstance(claims, list)
    assert len(claims) == 1


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


@pytest.mark.asyncio
async def test_faith_complete_called_with_correct_model_and_temperature() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _FAITHFUL_JSON})
    judge = FaithfulnessJudge(llm=llm)
    await judge.evaluate(_make_case(), _make_response())
    llm.complete.assert_called_once()
    kwargs: dict[str, Any] = llm.complete.call_args.kwargs
    assert kwargs["model"] == _MODEL
    assert kwargs["temperature"] == 0.0


# ===========================================================================
# RelevanceJudge
# ===========================================================================


@pytest.mark.asyncio
async def test_relevance_relevant_returns_1() -> None:
    judge = _make_relevance_judge(_RELEVANT_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.metric_name == "relevance"
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_relevance_irrelevant_returns_0() -> None:
    judge = _make_relevance_judge(_IRRELEVANT_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is None


@pytest.mark.asyncio
async def test_relevance_irrelevant_parts_in_details() -> None:
    judge = _make_relevance_judge(_IRRELEVANT_JSON)
    result = await judge.evaluate(_make_case(), _make_response())
    assert "irrelevant_parts" in result.details
    parts: Any = result.details["irrelevant_parts"]
    assert isinstance(parts, list)
    assert len(parts) == 1
    assert "NBA history" in parts[0]


@pytest.mark.asyncio
async def test_relevance_invalid_json_returns_error() -> None:
    judge = _make_relevance_judge("not json")
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "Invalid JSON" in result.error


@pytest.mark.asyncio
async def test_relevance_non_bool_relevant_returns_error() -> None:
    judge = _make_relevance_judge(
        '{"reasoning": "ok", "relevant": "true", "irrelevant_parts": []}'
    )
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None


# ===========================================================================
# CorrectnessJudge
# ===========================================================================


@pytest.mark.asyncio
async def test_correctness_score_4_returns_1() -> None:
    case = _make_case(expected_answer="Luka Doncic led with 33.9 PPG.")
    judge = _make_correctness_judge(_SCORE_4_JSON, _SCORE_4_JSON)
    result = await judge.evaluate(case, _make_response())
    assert result.metric_name == "correctness"
    assert result.value == pytest.approx(1.0)
    assert result.error is None


@pytest.mark.asyncio
async def test_correctness_score_2_returns_half() -> None:
    case = _make_case(expected_answer="Luka Doncic led with 33.9 PPG.")
    judge = _make_correctness_judge(_SCORE_2_JSON, _SCORE_2_JSON)
    result = await judge.evaluate(case, _make_response())
    assert result.value == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_correctness_score_0_returns_0() -> None:
    case = _make_case(expected_answer="Luka Doncic led with 33.9 PPG.")
    judge = _make_correctness_judge(_SCORE_0_JSON, _SCORE_0_JSON)
    result = await judge.evaluate(case, _make_response())
    assert result.value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_correctness_missing_expected_answer_returns_error() -> None:
    case = _make_case(expected_answer=None)
    judge = _make_correctness_judge(_SCORE_4_JSON)
    result = await judge.evaluate(case, _make_response())
    assert result.value == 0.0
    assert result.error is not None
    assert "expected answer" in result.error.lower()


@pytest.mark.asyncio
async def test_correctness_invalid_json_returns_error() -> None:
    case = _make_case(expected_answer="Luka led.")
    judge = _make_correctness_judge("not json", "not json")
    result = await judge.evaluate(case, _make_response())
    assert result.value == 0.0
    assert result.error is not None


@pytest.mark.asyncio
async def test_correctness_score_outside_range_returns_error() -> None:
    case = _make_case(expected_answer="Luka led.")
    bad_json = '{"reasoning": "too high", "score": 5, "errors": []}'
    judge = _make_correctness_judge(bad_json, bad_json)
    result = await judge.evaluate(case, _make_response())
    assert result.value == 0.0
    assert result.error is not None


@pytest.mark.asyncio
async def test_correctness_position_swap_calls_llm_twice() -> None:
    case = _make_case(expected_answer="Luka led.")
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": _SCORE_4_JSON})
    judge = CorrectnessJudge(llm=llm)
    await judge.evaluate(case, _make_response())
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_correctness_details_include_swap_fields() -> None:
    case = _make_case(expected_answer="Luka led.")
    judge = _make_correctness_judge(_SCORE_4_JSON, _SCORE_2_JSON)
    result = await judge.evaluate(case, _make_response())
    assert "forward_score" in result.details
    assert "swapped_score" in result.details
    assert "disagreement" in result.details
    assert "reasoning_forward" in result.details
    assert "reasoning_swapped" in result.details


@pytest.mark.asyncio
async def test_correctness_disagreement_computed_correctly() -> None:
    case = _make_case(expected_answer="Luka led.")
    fwd = '{"reasoning": "forward", "score": 4, "errors": []}'
    swp = '{"reasoning": "swapped", "score": 2, "errors": []}'
    judge = _make_correctness_judge(fwd, swp)
    result = await judge.evaluate(case, _make_response())
    assert result.details["forward_score"] == 4
    assert result.details["swapped_score"] == 2
    assert result.details["disagreement"] == 2
    assert result.value == pytest.approx((4 + 2) / 2.0 / 4.0)


# ===========================================================================
# RoutingJudge
# ===========================================================================


@pytest.mark.asyncio
async def test_routing_correct_route_returns_1() -> None:
    case = _make_case(question_type=QuestionType.FACTUAL)
    response = _make_response(routing_decision=QuestionType.FACTUAL)
    judge = RoutingJudge()
    result = await judge.evaluate(case, response)
    assert result.metric_name == "routing_accuracy"
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_routing_incorrect_route_returns_0() -> None:
    case = _make_case(question_type=QuestionType.FACTUAL)
    response = _make_response(routing_decision=QuestionType.ANALYTICAL)
    judge = RoutingJudge()
    result = await judge.evaluate(case, response)
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_routing_missing_decision_returns_0() -> None:
    case = _make_case(question_type=QuestionType.FACTUAL)
    response = _make_response(routing_decision=None)
    judge = RoutingJudge()
    result = await judge.evaluate(case, response)
    assert result.value == 0.0
    assert result.details["actual_route"] is None


@pytest.mark.asyncio
async def test_routing_details_include_routes() -> None:
    case = _make_case(question_type=QuestionType.HYBRID)
    response = _make_response(routing_decision=QuestionType.FACTUAL)
    judge = RoutingJudge()
    result = await judge.evaluate(case, response)
    assert result.details["expected_route"] == "hybrid"
    assert result.details["actual_route"] == "factual"


# ===========================================================================
# Calibration fixture tests
# ===========================================================================


def test_faith_calibration_has_10_cases() -> None:
    data: Any = yaml.safe_load(FAITH_FIXTURE.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 10


def test_faith_calibration_has_both_labels() -> None:
    data: Any = yaml.safe_load(FAITH_FIXTURE.read_text(encoding="utf-8"))
    labels: list[Any] = [c["human_label"] for c in data["cases"]]
    assert True in labels
    assert False in labels


def test_relevance_calibration_has_10_cases() -> None:
    data: Any = yaml.safe_load(RELEVANCE_FIXTURE.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 10


def test_relevance_calibration_has_both_labels() -> None:
    data: Any = yaml.safe_load(RELEVANCE_FIXTURE.read_text(encoding="utf-8"))
    labels: list[Any] = [c["human_label"] for c in data["cases"]]
    assert True in labels
    assert False in labels


def test_correctness_calibration_has_10_cases() -> None:
    data: Any = yaml.safe_load(CORRECTNESS_FIXTURE.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 10


def test_correctness_calibration_covers_all_scores() -> None:
    data: Any = yaml.safe_load(CORRECTNESS_FIXTURE.read_text(encoding="utf-8"))
    scores: set[int] = {c["human_score"] for c in data["cases"]}
    assert scores >= {0, 1, 2, 3, 4}


def test_routing_calibration_has_10_cases() -> None:
    data: Any = yaml.safe_load(ROUTING_FIXTURE.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 10


def test_routing_calibration_has_correct_and_incorrect() -> None:
    data: Any = yaml.safe_load(ROUTING_FIXTURE.read_text(encoding="utf-8"))
    correct_flags: list[Any] = [c["correct"] for c in data["cases"]]
    assert True in correct_flags
    assert False in correct_flags


# ===========================================================================
# Tool-use structured output tests
# ===========================================================================


def _tool_response(tool_name: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [{"id": "t1", "name": tool_name, "input": inp}],
    }


@pytest.mark.asyncio
async def test_faithfulness_accepts_tool_use_response() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _FAITHFULNESS_TOOL["name"],
            {"reasoning": "ok", "faithful": True, "unsupported_claims": []},
        )
    )
    judge = FaithfulnessJudge(llm=llm)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_faithfulness_passes_tool_schema() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _FAITHFULNESS_TOOL["name"],
            {"reasoning": "ok", "faithful": True, "unsupported_claims": []},
        )
    )
    judge = FaithfulnessJudge(llm=llm)
    await judge.evaluate(_make_case(), _make_response())
    kwargs = llm.complete.call_args.kwargs
    assert kwargs["tools"] == [_FAITHFULNESS_TOOL]
    assert kwargs["tool_choice"] == {"type": "tool", "name": _FAITHFULNESS_TOOL["name"]}


@pytest.mark.asyncio
async def test_relevance_accepts_tool_use_response() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _RELEVANCE_TOOL["name"],
            {"reasoning": "ok", "relevant": False, "irrelevant_parts": ["x"]},
        )
    )
    judge = RelevanceJudge(llm=llm)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.details["irrelevant_parts"] == ["x"]


@pytest.mark.asyncio
async def test_correctness_accepts_tool_use_response() -> None:
    case = _make_case(expected_answer="Luka led.")
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _CORRECTNESS_TOOL["name"],
            {"reasoning": "ok", "score": 4, "errors": []},
        )
    )
    judge = CorrectnessJudge(llm=llm)
    result = await judge.evaluate(case, _make_response())
    assert result.value == pytest.approx(1.0)
    assert result.details["forward_score"] == 4
    assert result.details["swapped_score"] == 4


@pytest.mark.asyncio
async def test_correctness_disagreement_flag_set_when_threshold_met() -> None:
    case = _make_case(expected_answer="Luka led.")
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[
            _tool_response(
                _CORRECTNESS_TOOL["name"],
                {"reasoning": "fwd", "score": 4, "errors": []},
            ),
            _tool_response(
                _CORRECTNESS_TOOL["name"],
                {"reasoning": "swp", "score": 1, "errors": ["x"]},
            ),
        ]
    )
    judge = CorrectnessJudge(llm=llm)
    result = await judge.evaluate(case, _make_response())
    assert result.details["disagreement"] == 3
    assert result.details["disagreement_flag"] is True


@pytest.mark.asyncio
async def test_correctness_disagreement_flag_unset_when_close() -> None:
    case = _make_case(expected_answer="Luka led.")
    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=[
            _tool_response(
                _CORRECTNESS_TOOL["name"],
                {"reasoning": "fwd", "score": 4, "errors": []},
            ),
            _tool_response(
                _CORRECTNESS_TOOL["name"],
                {"reasoning": "swp", "score": 3, "errors": []},
            ),
        ]
    )
    judge = CorrectnessJudge(llm=llm)
    result = await judge.evaluate(case, _make_response())
    assert result.details["disagreement"] == 1
    assert result.details["disagreement_flag"] is False


@pytest.mark.asyncio
async def test_faithfulness_rejects_malformed_tool_input() -> None:
    # score field missing / type-error → the judge should still report an error
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _FAITHFULNESS_TOOL["name"],
            {"reasoning": "bad", "faithful": "not-a-bool", "unsupported_claims": []},
        )
    )
    judge = FaithfulnessJudge(llm=llm)
    result = await judge.evaluate(_make_case(), _make_response())
    assert result.value == 0.0
    assert result.error is not None


@pytest.mark.asyncio
async def test_correctness_rejects_malformed_tool_input() -> None:
    case = _make_case(expected_answer="Luka led.")
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=_tool_response(
            _CORRECTNESS_TOOL["name"],
            {"reasoning": "bad", "score": 7, "errors": []},
        )
    )
    judge = CorrectnessJudge(llm=llm)
    result = await judge.evaluate(case, _make_response())
    assert result.value == 0.0
    assert result.error is not None


# ===========================================================================
# Routing judge: document the deterministic design
# ===========================================================================


def test_routing_prompt_file_is_documented_as_unused() -> None:
    """Routing judge is deterministic. The prompt file is a placeholder."""
    path = Path(__file__).parent.parent / "prompts" / "judges" / "routing" / "v1.txt"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # The file must explicitly document that it is not currently wired up.
    assert "not currently used" in content.lower() or "future use" in content.lower()
