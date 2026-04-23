import math

import pytest

from rageval.metrics.retrieval import (
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    ReciprocalRank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.types import Document, QuestionType, RAGResponse, TestCase

# ── Helpers ───────────────────────────────────────────────────────────────────


def _case(relevant_ids: list[str]) -> TestCase:
    return TestCase(
        id="case-1",
        question="Q?",
        question_type=QuestionType.FACTUAL,
        relevant_doc_ids=relevant_ids,
    )


def _response(doc_ids: list[str]) -> RAGResponse:
    docs = [Document(id=doc_id, content="text") for doc_id in doc_ids]
    return RAGResponse(answer="A", retrieved_docs=docs)


# ── precision_at_k ────────────────────────────────────────────────────────────


def test_precision_at_k_perfect() -> None:
    assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0


def test_precision_at_k_partial() -> None:
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)


def test_precision_at_k_zero_match() -> None:
    assert precision_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], ["a", "b"], 3) == 0.0


def test_precision_at_k_k_larger_than_retrieved() -> None:
    # only 2 docs retrieved, k=5; top-k is capped at len(retrieved), but k is the denominator
    assert precision_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(2 / 5)


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k(["a", "b"], ["a", "b"], 0) == 0.0


# ── recall_at_k ───────────────────────────────────────────────────────────────


def test_recall_at_k_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0


def test_recall_at_k_partial() -> None:
    assert recall_at_k(["a", "x", "b", "y"], ["a", "b", "c"], 4) == pytest.approx(2 / 3)


def test_recall_at_k_zero() -> None:
    assert recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["a", "b"], [], 2) == 0.0


def test_recall_at_k_empty_retrieved() -> None:
    assert recall_at_k([], ["a", "b"], 3) == 0.0


def test_recall_at_k_k_smaller_than_retrieved() -> None:
    # only the top-2 are considered; "b" at position 3 is excluded
    assert recall_at_k(["a", "x", "b"], ["a", "b"], 2) == pytest.approx(1 / 2)


# ── reciprocal_rank ───────────────────────────────────────────────────────────


def test_reciprocal_rank_first() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_second() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == pytest.approx(0.5)


def test_reciprocal_rank_third() -> None:
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match() -> None:
    assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


def test_reciprocal_rank_empty_retrieved() -> None:
    assert reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_empty_relevant() -> None:
    assert reciprocal_rank(["a", "b"], []) == 0.0


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────


def test_ndcg_at_k_perfect() -> None:
    assert ndcg_at_k(["a"], ["a"], 1) == 1.0


def test_ndcg_at_k_relevant_at_rank_2() -> None:
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0  → score = 1/log2(3)
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_at_k_relevant_at_rank_3() -> None:
    # DCG = 1/log2(4); IDCG = 1.0  → score = 1/log2(4)
    assert ndcg_at_k(["x", "y", "a"], ["a"], 3) == pytest.approx(1 / math.log2(4))


def test_ndcg_at_k_mixed_ranking() -> None:
    # retrieved: [a, x, b], relevant: [a, b]
    # DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(dcg / idcg)


def test_ndcg_at_k_no_relevant_retrieved() -> None:
    assert ndcg_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0


def test_ndcg_at_k_empty_relevant() -> None:
    assert ndcg_at_k(["a", "b"], [], 2) == 0.0


def test_ndcg_at_k_zero_k() -> None:
    assert ndcg_at_k(["a", "b"], ["a"], 0) == 0.0


def test_ndcg_at_k_k_larger_than_retrieved() -> None:
    # k=5 but only 1 doc; relevant at rank 1 → perfect score
    assert ndcg_at_k(["a"], ["a"], 5) == 1.0


def test_ndcg_at_k_perfect_multiple_relevant() -> None:
    # all top-k are relevant → 1.0
    assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == pytest.approx(1.0)


# ── Metric wrapper classes ────────────────────────────────────────────────────


def test_precision_at_k_wrapper_value() -> None:
    metric = PrecisionAtK(k=3)
    result = metric(_case(["a", "b"]), _response(["a", "x", "b"]))
    assert result.metric_name == "precision@3"
    assert result.case_id == "case-1"
    assert result.value == pytest.approx(2 / 3)


def test_recall_at_k_wrapper_value() -> None:
    metric = RecallAtK(k=3)
    result = metric(_case(["a", "b", "c"]), _response(["a", "b", "x"]))
    assert result.metric_name == "recall@3"
    assert result.case_id == "case-1"
    assert result.value == pytest.approx(2 / 3)


def test_reciprocal_rank_wrapper_value() -> None:
    metric = ReciprocalRank()
    result = metric(_case(["b"]), _response(["x", "b", "y"]))
    assert result.metric_name == "reciprocal_rank"
    assert result.case_id == "case-1"
    assert result.value == pytest.approx(0.5)


def test_ndcg_at_k_wrapper_value() -> None:
    metric = NDCGAtK(k=2)
    result = metric(_case(["a"]), _response(["a", "x"]))
    assert result.metric_name == "ndcg@2"
    assert result.case_id == "case-1"
    assert result.value == pytest.approx(1.0)


def test_precision_at_k_wrapper_no_relevant_docs() -> None:
    metric = PrecisionAtK(k=3)
    result = metric(_case([]), _response(["a", "b", "c"]))
    assert result.value == 0.0


def test_recall_at_k_wrapper_no_retrieved_docs() -> None:
    metric = RecallAtK(k=3)
    result = metric(_case(["a", "b"]), _response([]))
    assert result.value == 0.0


def test_reciprocal_rank_wrapper_no_retrieved_docs() -> None:
    metric = ReciprocalRank()
    result = metric(_case(["a"]), _response([]))
    assert result.value == 0.0


def test_ndcg_at_k_wrapper_no_retrieved_docs() -> None:
    metric = NDCGAtK(k=5)
    result = metric(_case(["a", "b"]), _response([]))
    assert result.value == 0.0
