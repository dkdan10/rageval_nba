"""Unit tests for scripts/calibrate_judge.py — fixture loading and agreement math."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.calibrate_judge import (
    binary_agreement,
    calibrate_faithfulness,
    calibrate_relevance,
    calibrate_routing,
    correctness_agreement,
    load_fixture,
)

# ── load_fixture ──────────────────────────────────────────────────────────────


def test_load_faithfulness_fixture_count() -> None:
    cases = load_fixture("faithfulness")
    assert len(cases) == 10


def test_load_faithfulness_fixture_fields() -> None:
    cases = load_fixture("faithfulness")
    for c in cases:
        assert "id" in c
        assert "question" in c
        assert "sources" in c
        assert "answer" in c
        assert "human_label" in c


def test_load_faithfulness_both_labels() -> None:
    cases = load_fixture("faithfulness")
    labels = [c["human_label"] for c in cases]
    assert True in labels
    assert False in labels


def test_load_relevance_fixture_count() -> None:
    assert len(load_fixture("relevance")) == 10


def test_load_relevance_fixture_fields() -> None:
    for c in load_fixture("relevance"):
        assert "question" in c
        assert "answer" in c
        assert "human_label" in c


def test_load_correctness_fixture_count() -> None:
    assert len(load_fixture("correctness")) == 10


def test_load_correctness_fixture_fields() -> None:
    for c in load_fixture("correctness"):
        assert "question" in c
        assert "answer" in c
        assert "expected_answer" in c
        assert "human_score" in c


def test_load_correctness_all_score_levels() -> None:
    scores = {c["human_score"] for c in load_fixture("correctness")}
    assert scores >= {0, 1, 2, 3, 4}


def test_load_routing_fixture_count() -> None:
    assert len(load_fixture("routing")) == 10


def test_load_routing_fixture_fields() -> None:
    for c in load_fixture("routing"):
        assert "question" in c
        assert "question_type" in c
        assert "routing_decision" in c
        assert "correct" in c


# ── binary_agreement ──────────────────────────────────────────────────────────


def test_binary_agreement_perfect() -> None:
    assert binary_agreement([True, True, False], [True, True, False]) == pytest.approx(1.0)


def test_binary_agreement_zero() -> None:
    assert binary_agreement([True, True], [False, False]) == pytest.approx(0.0)


def test_binary_agreement_partial() -> None:
    assert binary_agreement([True, False, True], [True, True, False]) == pytest.approx(1 / 3)


def test_binary_agreement_empty() -> None:
    assert binary_agreement([], []) == 0.0


def test_binary_agreement_all_true() -> None:
    assert binary_agreement([True, True, True], [True, True, True]) == pytest.approx(1.0)


# ── correctness_agreement ─────────────────────────────────────────────────────


def test_correctness_agreement_perfect() -> None:
    # value=1.0 → round(1.0 * 4) = 4 → matches human_score=4
    assert correctness_agreement([1.0, 0.0], [4, 0]) == pytest.approx(1.0)


def test_correctness_agreement_zero() -> None:
    # value=1.0 → score=4; human_score=0 → mismatch
    assert correctness_agreement([1.0, 1.0], [0, 0]) == pytest.approx(0.0)


def test_correctness_agreement_partial() -> None:
    # values: [1.0→4, 0.5→2, 0.0→0]; human: [4, 4, 0] → 2 matches
    assert correctness_agreement([1.0, 0.5, 0.0], [4, 4, 0]) == pytest.approx(2 / 3)


def test_correctness_agreement_empty() -> None:
    assert correctness_agreement([], []) == 0.0


def test_correctness_agreement_rounding() -> None:
    # value=0.625 → round(0.625 * 4) = round(2.5) = 2 (banker's rounding in Python)
    # human_score=3 → mismatch. value=0.75 → round(3.0)=3 → match.
    assert correctness_agreement([0.75], [3]) == pytest.approx(1.0)


# ── calibrate_faithfulness with mocked LLM ───────────────────────────────────


@pytest.mark.asyncio
async def test_calibrate_faithfulness_all_agree_faithful() -> None:
    # Judge always returns faithful=True.
    # Fixture: 5 faithful (human_label=True), 5 unfaithful (human_label=False).
    # Agreement = 5/10 = 0.5.
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={"content": '{"reasoning":"ok","faithful":true,"unsupported_claims":[]}'}
    )
    rate = await calibrate_faithfulness(llm=llm)
    assert rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_calibrate_faithfulness_all_agree_unfaithful() -> None:
    # Judge always returns faithful=False → agrees with 5 unfaithful cases.
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={
            "content": '{"reasoning":"bad","faithful":false,"unsupported_claims":["x"]}'
        }
    )
    rate = await calibrate_faithfulness(llm=llm)
    assert rate == pytest.approx(0.5)


# ── calibrate_relevance with mocked LLM ──────────────────────────────────────


@pytest.mark.asyncio
async def test_calibrate_relevance_all_relevant() -> None:
    # Judge always relevant=True; fixture has 5 relevant, 5 irrelevant → 50%.
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value={"content": '{"reasoning":"ok","relevant":true,"irrelevant_parts":[]}'}
    )
    rate = await calibrate_relevance(llm=llm)
    assert rate == pytest.approx(0.5)


# ── calibrate_routing (deterministic) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_calibrate_routing_perfect_agreement() -> None:
    # RoutingJudge is deterministic — it agrees with every fixture case by design.
    rate = await calibrate_routing()
    assert rate == pytest.approx(1.0)
