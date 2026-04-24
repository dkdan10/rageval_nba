"""Tests for Milestone 3.75: Router classifier (tool-use structured output)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rageval.demo.router import _MODEL, _TOOL_NAME, Router
from rageval.types import QuestionType


def _tool_response(category: str, reasoning: str = "test") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "tool_1",
                "name": _TOOL_NAME,
                "input": {"reasoning": reasoning, "category": category},
            }
        ],
    }


def _make_router(response: dict[str, Any]) -> Router:
    """Return a Router whose LLM always replies with *response*."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return Router(llm=llm)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_factual() -> None:
    router = _make_router(_tool_response("factual"))
    result = await router.classify("Who led the NBA in points per game in 2023-24?")
    assert result is QuestionType.FACTUAL


@pytest.mark.asyncio
async def test_classify_analytical() -> None:
    router = _make_router(_tool_response("analytical"))
    result = await router.classify("What are the four factors in basketball analytics?")
    assert result is QuestionType.ANALYTICAL


@pytest.mark.asyncio
async def test_classify_hybrid() -> None:
    router = _make_router(_tool_response("hybrid"))
    result = await router.classify(
        "What stats support Jokic being a historic offensive player, "
        "and how do analysts describe his game?"
    )
    assert result is QuestionType.HYBRID


@pytest.mark.asyncio
async def test_classify_unanswerable() -> None:
    router = _make_router(_tool_response("unanswerable"))
    result = await router.classify("Who will win MVP in 2027-28?")
    assert result is QuestionType.UNANSWERABLE


# ---------------------------------------------------------------------------
# Error / fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_calls_returns_unanswerable() -> None:
    router = _make_router({"content": "no tool call", "tool_calls": []})
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_missing_category_returns_unanswerable() -> None:
    router = _make_router(
        {
            "content": "",
            "tool_calls": [
                {"id": "t", "name": _TOOL_NAME, "input": {"reasoning": "ok"}}
            ],
        }
    )
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_unknown_category_returns_unanswerable() -> None:
    router = _make_router(_tool_response("unknown_value"))
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_non_dict_input_returns_unanswerable() -> None:
    router = _make_router(
        {
            "content": "",
            "tool_calls": [
                {"id": "t", "name": _TOOL_NAME, "input": "not an object"}
            ],
        }
    )
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_wrong_tool_name_returns_unanswerable() -> None:
    router = _make_router(
        {
            "content": "",
            "tool_calls": [
                {"id": "t", "name": "some_other_tool", "input": {"category": "factual"}}
            ],
        }
    )
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


# ---------------------------------------------------------------------------
# Call argument verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_called_with_tools() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_tool_response("factual"))
    router = Router(llm=llm)
    await router.classify("Who led PPG?")

    kwargs = llm.complete.call_args.kwargs
    assert kwargs["system"] == router._system
    assert kwargs["user"] == "Who led PPG?"
    assert kwargs["model"] == _MODEL
    assert kwargs["temperature"] == 0.0
    assert "tools" in kwargs
    tools = kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == _TOOL_NAME
    assert "input_schema" in tools[0]
    assert kwargs["tool_choice"] == {"type": "tool", "name": _TOOL_NAME}


@pytest.mark.asyncio
async def test_tool_schema_enumerates_all_categories() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_tool_response("factual"))
    router = Router(llm=llm)
    await router.classify("anything")

    tools = llm.complete.call_args.kwargs["tools"]
    enum = tools[0]["input_schema"]["properties"]["category"]["enum"]
    assert set(enum) == {qt.value for qt in QuestionType}


@pytest.mark.asyncio
async def test_router_loads_prompt_file() -> None:
    from rageval.demo.router import _PROMPT_PATH

    assert _PROMPT_PATH.exists()
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_tool_response("factual"))
    router = Router(llm=llm)
    assert router._system == _PROMPT_PATH.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_all_enum_members_reachable() -> None:
    for qt in QuestionType:
        router = _make_router(_tool_response(qt.value))
        result = await router.classify("test")
        assert result is qt


@pytest.mark.asyncio
async def test_result_is_enum_member() -> None:
    router = _make_router(_tool_response("factual"))
    result = await router.classify("test")
    assert isinstance(result, QuestionType)
    assert result == QuestionType.FACTUAL
    assert result.value == "factual"
