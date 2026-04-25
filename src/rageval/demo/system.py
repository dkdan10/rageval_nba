"""Reference hybrid RAG system for the NBA demo."""

from __future__ import annotations

import time
from typing import Protocol

from rageval.demo.rag_agent import RAGAgent
from rageval.demo.router import Router
from rageval.demo.sql_agent import SQLAgent
from rageval.demo.synthesizer import Synthesizer
from rageval.types import Document, QuestionType, RAGResponse, SQLResult


class _RouterLike(Protocol):
    async def classify(self, question: str) -> QuestionType: ...


class _SQLAgentLike(Protocol):
    async def generate_and_execute(self, question: str) -> SQLResult: ...


class _RAGAgentLike(Protocol):
    def retrieve(self, question: str, k: int = 5) -> list[Document]: ...


class _SynthesizerLike(Protocol):
    async def synthesize(
        self,
        question: str,
        sql_result: SQLResult | None = None,
        docs: list[Document] | None = None,
    ) -> str: ...


class HybridRAGSystem:
    name = "nba-hybrid-demo"

    def __init__(
        self,
        router: _RouterLike | None = None,
        sql_agent: _SQLAgentLike | None = None,
        rag_agent: _RAGAgentLike | None = None,
        synthesizer: _SynthesizerLike | None = None,
        top_k: int = 5,
    ) -> None:
        self.router = router or Router()
        self.sql_agent = sql_agent or SQLAgent()
        self.rag_agent = rag_agent or RAGAgent()
        self.synthesizer = synthesizer or Synthesizer()
        self.top_k = top_k

    async def answer(self, question: str) -> RAGResponse:
        started = time.perf_counter()
        errors: list[str] = []
        sql_result: SQLResult | None = None
        docs: list[Document] = []
        refused = False

        try:
            route = await self.router.classify(question)
        except Exception as exc:  # noqa: BLE001 - demo system should fail closed.
            route = QuestionType.UNANSWERABLE
            errors.append(f"router failed: {type(exc).__name__}: {exc}")

        if route is QuestionType.UNANSWERABLE:
            refused = True
            answer = "I cannot answer this question from the available NBA data sources."
            return RAGResponse(
                answer=answer,
                retrieved_docs=[],
                sql_result=None,
                routing_decision=route,
                latency_ms=_elapsed_ms(started),
                refused=refused,
            )

        if route in {QuestionType.FACTUAL, QuestionType.HYBRID}:
            try:
                sql_result = await self.sql_agent.generate_and_execute(question)
            except Exception as exc:  # noqa: BLE001
                error = f"SQL path failed: {type(exc).__name__}: {exc}"
                sql_result = SQLResult(
                    query="",
                    rows=[],
                    error=error,
                )
                errors.append(error)

        if route in {QuestionType.ANALYTICAL, QuestionType.HYBRID}:
            try:
                docs = self.rag_agent.retrieve(question, k=self.top_k)
            except Exception as exc:  # noqa: BLE001
                docs = []
                errors.append(f"RAG path failed: {type(exc).__name__}: {exc}")

        try:
            answer = await self.synthesizer.synthesize(question, sql_result, docs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"synthesizer failed: {type(exc).__name__}: {exc}")
            answer = "The system could not synthesize an answer from the available sources."

        if errors:
            answer = f"{answer} Errors: {'; '.join(errors)}"

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            sql_result=sql_result,
            routing_decision=route,
            latency_ms=_elapsed_ms(started),
            cost_usd=_component_cost(self.synthesizer),
            refused=refused,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _component_cost(component: object) -> float | None:
    llm = getattr(component, "_llm", None)
    cost = getattr(llm, "total_cost_usd", None)
    return float(cost) if isinstance(cost, int | float) else None
