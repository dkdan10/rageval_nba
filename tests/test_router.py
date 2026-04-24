"""Tests for Milestone 3.75: Router classifier."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rageval.demo.router import _MODEL, Router
from rageval.types import QuestionType


def _make_router(llm_content: str) -> Router:
    """Return a Router whose LLM always replies with *llm_content*."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": llm_content})
    return Router(llm=llm)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_factual() -> None:
    router = _make_router('{"reasoning": "stats lookup", "category": "factual"}')
    result = await router.classify("Who led the NBA in points per game in 2023-24?")
    assert result is QuestionType.FACTUAL


@pytest.mark.asyncio
async def test_classify_analytical() -> None:
    router = _make_router('{"reasoning": "conceptual definition", "category": "analytical"}')
    result = await router.classify("What are the four factors in basketball analytics?")
    assert result is QuestionType.ANALYTICAL


@pytest.mark.asyncio
async def test_classify_hybrid() -> None:
    router = _make_router('{"reasoning": "stats plus qualitative", "category": "hybrid"}')
    result = await router.classify(
        "What stats support Jokic being a historic offensive player, "
        "and how do analysts describe his game?"
    )
    assert result is QuestionType.HYBRID


@pytest.mark.asyncio
async def test_classify_unanswerable() -> None:
    router = _make_router('{"reasoning": "future prediction", "category": "unanswerable"}')
    result = await router.classify("Who will win MVP in 2027-28?")
    assert result is QuestionType.UNANSWERABLE


# ---------------------------------------------------------------------------
# Error / fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_returns_unanswerable() -> None:
    router = _make_router("not valid json at all")
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_missing_category_returns_unanswerable() -> None:
    router = _make_router('{"reasoning": "some reasoning"}')
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_unknown_category_returns_unanswerable() -> None:
    router = _make_router('{"reasoning": "weird", "category": "unknown_value"}')
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


@pytest.mark.asyncio
async def test_non_dict_json_returns_unanswerable() -> None:
    router = _make_router('"just a string"')
    result = await router.classify("anything")
    assert result is QuestionType.UNANSWERABLE


# ---------------------------------------------------------------------------
# Call argument verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_called_with_correct_args() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={"content": '{"reasoning": "stats", "category": "factual"}'}
    )
    router = Router(llm=llm)
    question = "Who led the NBA in points per game in 2023-24?"

    await router.classify(question)

    llm.complete.assert_called_once_with(
        system=router._system,
        user=question,
        model=_MODEL,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Enum member comparison (not string comparison)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_is_enum_member_not_string() -> None:
    router = _make_router('{"reasoning": "x", "category": "factual"}')
    result = await router.classify("test")
    assert isinstance(result, QuestionType)
    assert result == QuestionType.FACTUAL
    assert result.value == "factual"


@pytest.mark.asyncio
async def test_all_enum_members_reachable() -> None:
    for qt in QuestionType:
        router = _make_router(f'{{"reasoning": "test", "category": "{qt.value}"}}')
        result = await router.classify("test")
        assert result is qt


# ---------------------------------------------------------------------------
# Smoke test: mocked factual classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_factual_classification() -> None:
    router = _make_router('{"reasoning": "direct stat lookup", "category": "factual"}')
    result = await router.classify("Who led the NBA in points per game in 2023-24?")
    assert result is QuestionType.FACTUAL
