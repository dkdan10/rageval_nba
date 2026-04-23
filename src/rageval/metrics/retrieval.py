import math

from rageval.types import MetricResult, RAGResponse, TestCase


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0 or not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 if doc_id in relevant_set else 0.0) / math.log2(i + 2)
        for i, doc_id in enumerate(top_k)
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class PrecisionAtK:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        retrieved = [doc.id for doc in response.retrieved_docs]
        value = precision_at_k(retrieved, case.relevant_doc_ids, self.k)
        return MetricResult(
            metric_name=f"precision@{self.k}",
            case_id=case.id,
            value=value,
        )


class RecallAtK:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        retrieved = [doc.id for doc in response.retrieved_docs]
        value = recall_at_k(retrieved, case.relevant_doc_ids, self.k)
        return MetricResult(
            metric_name=f"recall@{self.k}",
            case_id=case.id,
            value=value,
        )


class ReciprocalRank:
    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        retrieved = [doc.id for doc in response.retrieved_docs]
        value = reciprocal_rank(retrieved, case.relevant_doc_ids)
        return MetricResult(
            metric_name="reciprocal_rank",
            case_id=case.id,
            value=value,
        )


class NDCGAtK:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, case: TestCase, response: RAGResponse) -> MetricResult:
        retrieved = [doc.id for doc in response.retrieved_docs]
        value = ndcg_at_k(retrieved, case.relevant_doc_ids, self.k)
        return MetricResult(
            metric_name=f"ndcg@{self.k}",
            case_id=case.id,
            value=value,
        )
